"""Fit the passive RLC network (circuit_model.Circuit) to a target S matrix.

    python train.py --target TARGET_S.npz [--seeds 0-15] [--out runs/name]

Initialisation per seed (circuit_model.Circuit): resonant branches drawn as (f0 log-uniform over the band, z, Q log-uniform);
DC-path edges metal or dielectric, with the A terminals grouped into conductors from the target's lowest frequency
(target_conductors).

A target marked grounded (parse_touchstone.py --grounded: 2n ports A1..An, B1..Bn, each against the solver's
reference) is fitted by the same network with grounded ports (Circuit(grounded=True)); every seed is then also scored
as the floating n-port (circuit_model.grounded_to_floating of the fit against that of the target).

Schedule (Adam, full frequency grid every step, parameters projected into the element-value window after every step):
  1. DC-path pretrain: resonant block removed, only the DC-path parameters trained, on the points f <= --pretrain-fmax,
     relative error, --pretrain-steps steps, lr --pretrain-lr cosine-annealed to 0.
  2. Joint training, coarse: all parameters, relative error, the first --switch-frac of the --steps joint steps.
  3. Joint training, fine: all parameters, complex-log error, the remaining steps.
  Stages 2-3 share one one-cycle lr schedule (peak --lr). Several seeds are trained at once (torch.func.vmap over the
  stacked parameters; on a GPU each step is one CUDA-graph replay).

Loss, per frequency, summed over H in {S, Z/z0, Y z0} of the mean over the matrix entries; with r = X / Xt:
  relative error     |X - Xt| / d,  d = |Xt| + phi max_ij |Xt|, floor phi = --floor-s for S, --floor-zy for Z, Y
  complex-log error  |ln r| = sqrt(ln^2 |r| + arg^2 r), over the entries that are not exactly 0 in the target
The training loss is the mean over the frequencies. The Z, Y targets are derived from the S target; where one does
not exist (e.g. Z at DC of isolated conductors) it is left out at that frequency.

Outputs in --out: summary.md (target, settings, score statistics, one row per seed), summary.json (the same plus the
full metrics and loss curve of every seed, speed, timings), and for the best seed best/model.pt (checkpoint),
best/model.cir (SPICE subcircuit), best/bode_{S,Z,Y}.png, best/loss.png. --save-all also writes
seeds/model_s<seed>.{pt,cir} for every seed.
"""
import argparse
import json
import math
import os
import time

import numpy as np
import torch
from torch.func import functional_call, vmap

import circuit_model as cm
from circuit_model import Circuit
from utils.metrics import fit_metrics, one_line, score
from utils.plotting import plot_bode, plot_loss
from utils.wideband import describe, plot_wide, wide_sweep
from slim import slim_circuit


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', required=True, help='S-parameter .npz (freq, params, z0) from parse_touchstone.py')
    ap.add_argument('--out', default='runs/train', help='output directory')
    ap.add_argument('--seeds', default='0', help="one seed '3', a range '0-15' or a list '0,3,5' (trained batched)")
    # topology and element-value window
    ap.add_argument('--rlc-nodes', type=int, default=16, help='private nodes of the resonant block')
    ap.add_argument('--dc-nodes', type=int, default=0, help='private nodes of the DC path')
    ap.add_argument('--no-dc-path', action='store_true',
                    help='remove the DC path (RL block: no private nodes, no edges) and its pretrain')
    ap.add_argument('--dc-fl-max', type=float, default=1e6, help='max corner R/(2 pi L) of the DC-path edges (Hz)')
    ap.add_argument('--l-max', type=float, default=1.0, help='max inductance (H)')
    ap.add_argument('--c-max', type=float, default=1e-3, help='max capacitance (F)')
    ap.add_argument('--r-min', type=float, default=1e-2, help='min resistance (Ohm)')
    ap.add_argument('--r-max', type=float, default=1e15, help='max resistance (Ohm)')
    ap.add_argument('--q-max', type=float, default=30.0, help='max quality factor of a resonant branch')
    ap.add_argument('--leak', type=float, default=1e-12, help='leakage conductance from every node to GND (S)')
    # loss
    ap.add_argument('--floor-s', type=float, default=0.1, help='relative-error floor phi for S (x max |S_ij|)')
    # out-of-band penalty
    ap.add_argument('--oob-weight', type=float, default=0.1,
                    help='weight of the out-of-band curvature penalty (0: off)')
    ap.add_argument('--oob-fmin', type=float, default=1.0, help='low end of the out-of-band range (Hz)')
    ap.add_argument('--oob-fmax', type=float, default=1e10,
                    help='high end of the out-of-band range (Hz); the target band top: penalty below the band only')
    ap.add_argument('--check-fmin', type=float, default=1.0, help='wide-band check of every seed: low end (Hz)')
    ap.add_argument('--check-fmax', type=float, default=1e10, help='wide-band check of every seed: high end (Hz)')
    ap.add_argument('--oob-margin', type=float, default=3.0,
                    help='frequency factor on each side of the band left out of the penalty and of the out-of-band '
                         'report (f_lo / margin .. f_lo, f_hi .. f_hi margin): room for resonances just outside the '
                         'band whose tails reach into it')
    ap.add_argument('--oob-points', type=int, default=32, help='random frequency triplets per step')
    ap.add_argument('--oob-kappa', type=float, default=1.0,
                    help='curvature |d2 ln H / d(ln f)2| allowed out of band (1: a critically damped pole pair)')
    ap.add_argument('--floor-zy', type=float, default=1e-3, help='relative-error floor phi for Z/z0 and Y z0 (x max)')
    # schedule
    ap.add_argument('--pretrain-steps', type=int, default=8000)
    ap.add_argument('--pretrain-fmax', type=float, default=1.05e7, help='DC-path pretrain on the points f <= this (Hz)')
    ap.add_argument('--pretrain-lr', type=float, default=0.1)
    ap.add_argument('--steps', type=int, default=30000, help='joint-training steps')
    ap.add_argument('--switch-frac', type=float, default=0.3,
                    help='fraction of the joint steps trained with the relative error before the complex-log error')
    ap.add_argument('--lr', type=float, default=0.02, help='peak lr of the one-cycle schedule')
    ap.add_argument('--slim-tol', type=float, default=0.05,
                    help='after training, remove every element whose removal raises no error component by more than '
                         'this (dB; slim.py), then fine-tune; 0: no slimming')
    ap.add_argument('--finetune-frac', type=float, default=1 / 3,
                    help='the fine-tune after slimming repeats this last fraction of the joint schedule')
    ap.add_argument('--pct-start', type=float, default=0.1, help='one-cycle warm-up fraction')
    ap.add_argument('--lr-final-div', type=float, default=800.0, help='one-cycle final_div_factor')
    # fine-tuning (e.g. after slim.py)
    ap.add_argument('--init', default=None,
                    help='start from this checkpoint (parameters and element masks) instead of a random init and '
                         'skip the pretrain; may contain {seed}')
    ap.add_argument('--start-step', type=int, default=0, help='start the joint schedule at this step (with --init)')
    # misc
    ap.add_argument('--deterministic', action='store_true',
                    help='deterministic GPU kernels (bit-identical reruns, ~1.3x slower)')
    ap.add_argument('--save-all', action='store_true', help='also save the checkpoint and netlist of every seed')
    ap.add_argument('--quiet', action='store_true', help='no plots')
    return ap.parse_args(argv)


def parse_seeds(text):
    seeds = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        seeds += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return seeds


def target_conductors(freqs, S, k_min=0.5):
    """Groups of ports whose A terminals lie on one conductor, from the target's lowest frequency (DC if present,
    where conduction along a conductor dominates any capacitive coupling): ports p, q share a conductor when the
    normalized coupling |Y_pq| / sqrt(|Y_pp Y_qq|) exceeds k_min (1 for the two ends of one trace, ~0 between
    isolated conductors). None when Y does not exist there."""
    S0 = S[int(np.argmin(freqs))]
    n = S0.shape[-1]
    try:
        Y0 = np.linalg.solve(np.eye(n) + S0, np.eye(n) - S0)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(Y0).all():
        return None
    d = np.sqrt(np.abs(np.diag(Y0)))
    k = np.abs(Y0) / (np.outer(d, d) + 1e-300)
    group = list(range(n))
    for p in range(n):
        for q in range(p + 1, n):
            if k[p, q] > k_min:
                old, new = group[q], group[p]
                group = [new if g == old else g for g in group]
    return [[p for p in range(n) if group[p] == g] for g in sorted(set(group))]


class Loss:
    """Relative and complex-log errors of S, Z/z0, Y z0 against fixed targets (see the module docstring). A Z or Y
    target that does not exist at a frequency (cm.target_normalized: NaN, e.g. Z at DC of isolated conductors) is left
    out of the loss there."""

    def __init__(self, S_t, floor_s, floor_zy, freq=None, z0=50.0):
        self.terms = []
        for kind, H, floor in (('S', S_t, floor_s), ('Z', cm.target_normalized(S_t, 'Z', freq, z0), floor_zy),
                               ('Y', cm.target_normalized(S_t, 'Y', freq, z0), floor_zy)):
            valid = torch.isfinite(H).all(dim=(-2, -1))
            H = torch.where(valid[:, None, None], H, torch.ones_like(H))
            ref = H.abs()
            den = ref + floor * ref.amax(dim=(-2, -1), keepdim=True)
            # an entry exactly 0 in the target (e.g. the coupling of conductors that a solver separates exactly at DC)
            # has no logarithm: left out of the complex-log error, kept in the relative error
            nz = (ref > 0).to(ref.dtype)
            self.terms.append((kind, H, den, torch.log(ref + 1e-300), valid.to(ref.dtype), nz))

    def _preds(self, S):
        return {'S': S, 'Z': cm.s_to_normalized(S, 'Z'), 'Y': cm.s_to_normalized(S, 'Y')}

    def __call__(self, S, kind):
        """Per-frequency loss (N_freq,) of the prediction S; kind 'rel' (relative) or 'log' (complex log)."""
        X = self._preds(S)
        out = 0
        for k, H, den, log_ref, valid, nz in self.terms:
            if kind == 'rel':
                e = ((X[k] - H).abs() / den).mean(dim=(-2, -1))
            else:
                lr = torch.log(X[k].abs() + 1e-300) - log_ref
                ang = torch.angle(X[k] * H.conj())
                e = (torch.sqrt(lr ** 2 + ang ** 2 + 1e-30) * nz).sum(dim=(-2, -1)) / nz.sum(dim=(-2, -1)).clamp(min=1)
            out = out + e * valid
        return out


class OutOfBand:
    """Penalty on resonant behaviour outside the target band [f_lo, f_hi], where there is no target.

    Every step draws n frequency triplets (f e^-h, f, f e^h), f log-uniform over [fmin, f_lo / margin] and
    [f_hi margin, fmax] (margin, a frequency factor: room for resonances just outside the band, whose tails reach into it), and
    measures the curvature of the model's self-immittances,
    kappa = |d2 ln H / d(ln f)2|, H in {diag Z/z0, diag Y z0}, by a second difference. A single real pole or zero bends
    ln H by at most 1/2; a critically damped pair (Q = 1/2) by 1; a resonance of quality Q by ~Q^2. The penalty is the
    mean of max(kappa - kappa_max, 0): smooth extrapolation (R, L, C behaviour, real corners) is free, peaks are not.
    The triplets are redrawn in place before every step (the CUDA graph reads self.freq), so no frequency is blind."""

    def __init__(self, f_lo, f_hi, fmin, fmax, n, h=0.01, kappa=1.0, margin=3.0):
        self.n, self.h, self.kappa = n, h, kappa
        m = math.log(margin)   # left free on each side of the band: f_lo / margin .. f_lo, f_hi .. f_hi margin
        segs = [(math.log(fmin) + h, math.log(f_lo) - m - h), (math.log(f_hi) + m + h, math.log(fmax) - h)]
        self.segs = [(a, b) for a, b in segs if b > a]
        if not self.segs:
            raise ValueError('out-of-band penalty: no range outside the target band')
        self.widths = torch.tensor([b - a for a, b in self.segs], dtype=cm.DTYPE, device=cm.DEVICE)
        self.starts = torch.tensor([a for a, _ in self.segs], dtype=cm.DTYPE, device=cm.DEVICE)
        self.freq = torch.empty(3 * n, dtype=cm.DTYPE, device=cm.DEVICE)
        self.resample()

    def describe(self):
        return ' and '.join(f'{math.exp(a - self.h):.3g}-{math.exp(b + self.h):.3g} Hz' for a, b in self.segs)

    def resample(self):
        x = torch.rand(self.n, dtype=cm.DTYPE, device=cm.DEVICE) * self.widths.sum()
        cum = torch.cumsum(self.widths, 0)
        i = torch.searchsorted(cum, x, right=True).clamp(max=len(self.segs) - 1)
        u = self.starts[i] + x - (cum[i] - self.widths[i])
        self.freq.copy_(torch.exp(torch.stack([u - self.h, u, u + self.h], 1)).flatten())

    def curvature(self, S):
        """(n, 2 ports) curvatures of diag Z/z0 and diag Y z0 from the model's S at self.freq."""
        out = []
        for kind in 'ZY':
            H = torch.diagonal(cm.s_to_normalized(S, kind), dim1=-2, dim2=-1).reshape(self.n, 3, -1)
            out.append((torch.log(H[:, 2] / H[:, 1]) - torch.log(H[:, 1] / H[:, 0])).abs() / self.h ** 2)
        return torch.cat(out, -1)

    def __call__(self, S):
        return torch.relu(self.curvature(S) - self.kappa).mean()


def capture_step(closure, params):
    """On a GPU: capture closure() (which writes p.grad for every p in params) in a CUDA graph and return its replay
    (the step is launch-bound: one replay instead of ~1-2k kernel launches). On a CPU: the closure itself."""
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    if cm.DEVICE != 'cuda':
        return closure
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        for _ in range(3):   # warm-up: allocator, cuBLAS/cuSOLVER handles
            closure()
    torch.cuda.current_stream().wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        out = closure()

    def replay():
        graph.replay()
        return out
    replay.closure = closure   # keeps every tensor the graph reads (e.g. the closure's inputs) alive
    return replay


def main(argv=None):
    args = parse_args(argv)
    if args.deterministic:
        cm.set_deterministic()
    os.makedirs(args.out, exist_ok=True)
    seeds = parse_seeds(args.seeds)
    K = len(seeds)

    data = np.load(args.target)
    freqs_np, S_np, z0 = data['freq'], data['params'], float(data['z0'])
    grounded = bool(data['grounded']) if 'grounded' in data else False
    n_ports = S_np.shape[1] // 2 if grounded else S_np.shape[1]
    pos = freqs_np[freqs_np > 0]
    f_lo, f_hi = float(pos.min()), float(pos.max())
    freqs = torch.tensor(freqs_np, dtype=cm.DTYPE, device=cm.DEVICE)
    S_t = torch.tensor(S_np, dtype=cm.CDTYPE, device=cm.DEVICE)
    loss_fn = Loss(S_t, args.floor_s, args.floor_zy, freqs, z0)
    print(f'target {args.target}: {S_np.shape[1]} {"grounded" if grounded else "floating"} ports, {len(freqs_np)} points ({f_lo:.4g}-{f_hi:.4g} Hz'
          f'{", plus DC" if (freqs_np == 0).any() else ""}), z0 {z0:g} Ohm; device {cm.DEVICE}')

    # one Circuit per seed (same topology, own init); trained stacked (K, ...) through vmap over the first one
    window = dict(rlc_nodes=args.rlc_nodes, dc_nodes=args.dc_nodes, dc_fl_max=args.dc_fl_max, l_max=args.l_max,
                  c_max=args.c_max, r_min=args.r_min, r_max=args.r_max, q_max=args.q_max, leak=args.leak,
                  dc_path=not args.no_dc_path, grounded=grounded)
    # floating view of a grounded target (the conductors of the A terminals are read from it)
    S_fl_t = cm.grounded_to_floating(S_t, z0).cpu() if grounded else None
    conductors = target_conductors(freqs_np, S_fl_t.numpy() if grounded else S_np)
    print(f'conductors of the A terminals (initial DC path, from {freqs_np.min():g} Hz): {fmt_conductors(conductors)}')
    circuits = []
    for sd in seeds:
        torch.manual_seed(sd)
        c = Circuit(n_ports, f_lo, f_hi, conductors=conductors, **window)
        if args.init:
            c, _ = cm.load_checkpoint(args.init.format(seed=sd))
        circuits.append(c)
    base = circuits[0]
    print(base.summary())
    print(f'seeds {seeds} ({K} model{"s" * (K > 1)} batched)')
    names = [n for n, _ in base.named_parameters()]
    P = {n: torch.stack([dict(c.named_parameters())[n].detach() for c in circuits]).clone().requires_grad_()
         for n in names}
    masks = {b: torch.stack([getattr(c, b) for c in circuits]) for b in ('branch_alive', 'dc_alive')}

    def project():
        with torch.no_grad():
            base.project_params_(P)

    def model_S(p, al, f, rlc_off=False):
        return functional_call(base, {**p, **al}, (f,), {'z0': z0, 'rlc_off': rlc_off})

    def make_closure(params, f, loss, kind, rlc_off=False, oob=None):
        """Gradient of the `kind` loss (+ oob_weight x the out-of-band penalty, if oob) for every model: writes p.grad,
        returns the (K,) losses and (K,) penalties."""
        def one(p, al):
            data = loss(model_S(p, al, f, rlc_off), kind).mean()
            pen = oob(model_S(p, al, oob.freq)) if oob is not None else torch.zeros_like(data)
            return data, pen

        def closure():
            losses, pens = vmap(one)(P, masks)
            total = losses.sum() + args.oob_weight * pens.sum()
            for p, g in zip(params, torch.autograd.grad(total, params)):
                p.grad.copy_(g)
            return losses.detach(), pens.detach()
        return closure

    t_start = time.time()
    pretrain_loss = [float('nan')] * K
    # 1. DC-path pretrain
    if not args.init and args.pretrain_steps > 0 and base.n_dc:
        dc_params = [P[n] for n in ('dc_logR', 'dc_logL')]
        idx = torch.nonzero(freqs <= args.pretrain_fmax).flatten()
        print(f'DC-path pretrain: {len(idx)} points (f <= {args.pretrain_fmax:g} Hz), {args.pretrain_steps} steps')
        pre_loss = Loss(S_t[idx], args.floor_s, args.floor_zy, freqs[idx], z0)
        step = capture_step(make_closure(dc_params, freqs[idx], pre_loss, 'rel', rlc_off=True), dc_params)
        opt = torch.optim.Adam(dc_params, lr=args.pretrain_lr, fused=cm.DEVICE == 'cuda')
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.pretrain_steps)
        t0 = time.time()
        for _ in range(args.pretrain_steps):
            losses, _ = step()
            opt.step()
            sch.step()
            project()
        pretrain_loss = losses.cpu().tolist()
        print(f'  loss {fmt_k(losses)}  ({args.pretrain_steps / (time.time() - t0):.0f} it/s)')
    t_pre = time.time()

    # 2-3. joint training: relative error until the switch step, complex-log error after it
    plist = [P[n] for n in names]
    switch = round(args.switch_frac * args.steps)
    oob = None
    if args.oob_weight > 0:
        oob = OutOfBand(f_lo, f_hi, args.oob_fmin, args.oob_fmax, args.oob_points, kappa=args.oob_kappa,
                        margin=args.oob_margin)
        print(f'out-of-band penalty: weight {args.oob_weight:g}, curvature above {args.oob_kappa:g}, '
              f'{args.oob_points} random triplets per step in {oob.describe()}')
    step_rel = capture_step(make_closure(plist, freqs, loss_fn, 'rel', oob=oob), plist)
    step_log = capture_step(make_closure(plist, freqs, loss_fn, 'log', oob=oob), plist)
    log_every = max(1, args.steps // 20)

    def run_joint(first):
        """Steps first .. steps-1 of the one-cycle schedule (fresh Adam); returns the (steps, K) loss and penalty
        curves and the rate (it/s)."""
        opt = torch.optim.Adam(plist, lr=args.lr, fused=cm.DEVICE == 'cuda')
        sch = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps,
                                                  pct_start=args.pct_start, div_factor=10,
                                                  final_div_factor=args.lr_final_div)
        for _ in range(first):
            sch.step()
        curve = torch.zeros(args.steps - first, K, dtype=cm.DTYPE, device=cm.DEVICE)
        pen_curve = torch.zeros_like(curve)
        t0 = time.time()
        for i in range(first, args.steps):
            if oob is not None:
                oob.resample()
            losses, pens = step_rel() if i < switch else step_log()
            opt.step()
            sch.step()
            project()
            curve[i - first] = losses
            pen_curve[i - first] = pens
            if i % log_every == 0 or i == args.steps - 1:
                print(f'step {i:6d}  {"rel" if i < switch else "log"} loss {fmt_k(losses)}'
                      + (f'  out-of-band {fmt_k(pens)}' if oob is not None else '') + f'  lr {sch.get_last_lr()[0]:.2e}')
        if cm.DEVICE == 'cuda':
            torch.cuda.synchronize()
        return curve.cpu(), pen_curve.cpu(), (args.steps - first) / (time.time() - t0)

    def to_circuits():
        with torch.no_grad():
            for k, c in enumerate(circuits):
                for n, p in c.named_parameters():
                    p.copy_(P[n][k])

    print(f'joint training: {args.steps} steps, relative error until step {switch}, then complex-log error')
    t_setup = time.time()
    curve, pen_curve, rate = run_joint(args.start_step)
    t_joint = time.time()
    print(f'joint training: {rate:.0f} it/s x {K} models; {t_joint - t_setup:.0f} s')

    # 4-5. slim every seed (slim.py), then fine-tune the slimmed networks over the last --finetune-frac of the schedule
    n_full = [int(c.branch_alive.sum()) + int(c.dc_alive.sum()) for c in circuits]
    ft_curve = ft_pen = None
    t_slim = t_ft = t_joint
    if args.slim_tol > 0:
        to_circuits()
        freqs_cpu, S_t_cpu = freqs.cpu(), S_t.cpu()
        extra_cpu = {k: cm.target_normalized(S_t_cpu, k, freqs_cpu, z0) for k in 'ZY'}
        for k, (sd, c) in enumerate(zip(seeds, circuits)):
            n0, n1, s0, s1 = slim_circuit(c, freqs, S_t_cpu, extra_cpu, z0, args.slim_tol, log=lambda m: None)
            print(f'slim seed {sd}: {n0} -> {n1} elements, score {s0:.3f} -> {s1:.3f} dB')
            masks['branch_alive'][k].copy_(c.branch_alive)
            masks['dc_alive'][k].copy_(c.dc_alive)
        t_slim = time.time()
        first = round((1 - args.finetune_frac) * args.steps)
        print(f'fine-tune of the slimmed networks: steps {first} to {args.steps}')
        ft_curve, ft_pen, ft_rate = run_joint(first)
        t_ft = time.time()
        print(f'slim {t_slim - t_joint:.0f} s, fine-tune {ft_rate:.0f} it/s x {K} models, {t_ft - t_slim:.0f} s')
    t_end = time.time()
    print(f'total {t_end - t_start:.0f} s')

    # outputs: every seed scored in summary.{md,json}; checkpoint, netlist and plots of the best seed in best/
    to_circuits()
    runs, preds = [], []
    freqs_cpu, S_t_cpu = freqs.cpu(), S_t.cpu()
    extra = {k: cm.target_normalized(S_t_cpu, k, freqs_cpu, z0) for k in 'ZY'}
    extra_fl = {k: cm.target_normalized(S_fl_t, k, freqs_cpu, z0) for k in 'ZY'} if grounded else None
    for k, (sd, c) in enumerate(zip(seeds, circuits)):
        with torch.no_grad():
            S = c(freqs, z0=z0)
            final = loss_fn(S, 'log').mean().item()
        preds.append(S.cpu())
        m = fit_metrics(freqs_cpu, S_t_cpu, preds[-1], extra)
        print(f'seed {sd}: score {score(m):.3f} dB  log loss {final:.4e}  {one_line(m)}')
        m_fl = fit_metrics(freqs_cpu, S_fl_t, cm.grounded_to_floating(preds[-1], z0), extra_fl) if grounded else None
        if grounded:
            print(f'   as floating {n_ports}-port: score {score(m_fl):.3f} dB  {one_line(m_fl)}')
        passive = cm.check_passivity(c, z0=z0, extra_freqs=freqs)
        w = wide_sweep(c, z0, f_lo, f_hi, args.check_fmin, args.check_fmax, margin=args.oob_margin)
        print(f'   wide band {args.check_fmin:g}-{args.check_fmax:g} Hz: {describe(w)}')
        if args.save_all:
            save_model(c, sd, args, z0, os.path.join(args.out, 'seeds', f'model_s{sd}'))
        runs.append({'seed': sd, 'score_db': score(m), 'final_log_loss': final, 'pretrain_loss': pretrain_loss[k],
                     'passive': passive, 'metrics': m,
                     'score_floating_db': score(m_fl) if grounded else None, 'metrics_floating': m_fl,
                     'wide': {'kappa_max': w['kappa_max'], 'peaks': w['peaks'], 'Z_dc_exists': w['Z_dc_exists']},
                     'loss_curve': curve[::max(1, len(curve) // 100), k].tolist(),
                     'oob_curve': pen_curve[::max(1, len(curve) // 100), k].tolist() if oob is not None else None,
                     'elements_full': n_full[k], 'elements': int(c.branch_alive.sum()) + int(c.dc_alive.sum()),
                     'finetune_curve': ft_curve[::max(1, len(ft_curve) // 100), k].tolist()
                     if ft_curve is not None else None})
    s = np.array([r['score_db'] for r in runs])
    kb = int(s.argmin())
    print(f'score over {K} seed(s): mean {s.mean():.3f}  std {s.std(ddof=1) if K > 1 else 0:.3f}  '
          f'median {np.median(s):.3f}  best {s[kb]:.3f} (seed {seeds[kb]})')
    best_dir = os.path.join(args.out, 'best')
    save_model(circuits[kb], seeds[kb], args, z0, os.path.join(best_dir, 'model'))
    if not args.quiet:
        plot_bode(freqs_cpu, S_t_cpu, preds[kb], os.path.join(best_dir, 'bode_S.png'))
        for kind, H in extra.items():
            plot_bode(freqs_cpu, H, cm.s_to_normalized(preds[kb], kind), os.path.join(best_dir, f'bode_{kind}.png'),
                      kind=f'{kind}n')
        if grounded:   # the floating n-port of the fit against that of the target
            S_fl = cm.grounded_to_floating(preds[kb], z0)
            plot_bode(freqs_cpu, S_fl_t, S_fl, os.path.join(best_dir, 'bode_floating_S.png'))
            for kind, H in extra_fl.items():
                plot_bode(freqs_cpu, H, cm.s_to_normalized(S_fl, kind),
                          os.path.join(best_dir, f'bode_floating_{kind}.png'), kind=f'{kind}n')
        plot_loss(curve[:, kb].tolist(), os.path.join(best_dir, 'loss.png'), switch=switch - args.start_step)
        H_t = {k: torch.diagonal(H, dim1=-2, dim2=-1).numpy() for k, H in extra.items()}
        plot_wide(wide_sweep(circuits[kb], z0, f_lo, f_hi, args.check_fmin, args.check_fmax,
                             margin=args.oob_margin),
                  os.path.join(best_dir, 'wide.png'), freqs_np, H_t, f_lo, f_hi, f'seed {seeds[kb]}')
    info = {'args': vars(args), 'K': K, 'best_seed': seeds[kb], 'conductors': conductors, 'grounded': grounded,
            'device': torch.cuda.get_device_name() if cm.DEVICE == 'cuda' else 'cpu',
            'torch': torch.__version__, 'deterministic': torch.are_deterministic_algorithms_enabled(),
            'joint_it_per_s': rate, 'timing_s': {'pretrain': t_pre - t_start, 'graph_capture': t_setup - t_pre,
                                                 'joint': t_joint - t_setup, 'slim': t_slim - t_joint,
                                                 'finetune': t_ft - t_slim, 'total': t_end - t_start}}
    with open(os.path.join(args.out, 'summary.json'), 'w') as fh:
        json.dump({**info, 'runs': runs}, fh, indent=1)
    write_summary_md(os.path.join(args.out, 'summary.md'), info, runs, freqs_np, n_ports, z0, base)
    return runs


def save_model(c, seed, args, z0, stem):
    """stem.pt (checkpoint: config, parameters, element masks) and stem.cir (SPICE subcircuit)."""
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    torch.save({'config': c.config, 'state_dict': c.state_dict(), 'seed': seed, 'args': vars(args)}, stem + '.pt')
    c.write_spice(stem + '.cir', z0=z0)


def write_summary_md(path, info, runs, freqs, n_ports, z0, base):
    """Human-readable run summary: target, settings, score statistics, per-seed table, files."""
    a = info['args']
    s = np.array([r['score_db'] for r in runs])
    K, best = len(runs), info['best_seed']
    pos = freqs[freqs > 0]
    band = lambda r, q: np.mean([v[q] for v in r['metrics'].values()])
    defaults = vars(parse_args(['--target', a['target']]))
    changed = {k: v for k, v in a.items()
               if k not in ('target', 'out', 'seeds', 'save_all', 'quiet', 'deterministic') and v != defaults[k]}
    flag = lambda k, v: f'`--{k.replace("_", "-")} {v:g}`' if isinstance(v, float) else f'`--{k.replace("_", "-")} {v}`'
    q = np.percentile(s, [5, 95]) if K > 1 else (s[0], s[0])
    lines = [
        f'# Fit summary: {os.path.basename(a["target"])}', '',
        f'- **Target:** `{a["target"]}`, ' + (f'{2 * n_ports} grounded ports (terminals A1..A{n_ports}, B1..B{n_ports}, '
                                               f'each against GND)' if info['grounded'] else f'{n_ports} floating ports')
        + f', {len(freqs)} points '
        f'({pos.min():.4g} Hz to {pos.max():.4g} Hz{", plus DC" if (freqs == 0).any() else ""}), z0 = {z0:g} Ohm.',
        f'- **Model:** {a["rlc_nodes"]} resonant-block nodes ({base.n_rlc} series-RLC branches); ' + (
            f'{base.dc_nodes} DC-path nodes ({base.n_dc} series-RL edges); conductors of the A terminals (initial DC '
            f'path, from {freqs.min():g} Hz): {fmt_conductors(info["conductors"])}.' if base.n_dc else 'no DC path.'),
        f'- **Settings:** {"defaults" if not changed else "defaults except " + ", ".join(flag(k, v) for k, v in changed.items())}.',
        f'- **Seeds:** {a["seeds"]} ({K}), trained batched on {info["device"]} in {info["timing_s"]["total"]:.0f} s'
        f'{" (deterministic)" if info["deterministic"] else ""}: ' + ', '.join(
            f'{name} {info["timing_s"][key]:.0f} s' for name, key in (
                ('pretrain', 'pretrain'), ('graph capture', 'graph_capture'), ('joint training', 'joint'),
                ('slimming', 'slim'), ('fine-tune', 'finetune'))) + '.',
        (f'- **Slimming:** elements whose removal raises no error component by more than {a["slim_tol"]:g} dB are '
         f'removed, then the last {a["finetune_frac"]:.2g} of the schedule is repeated; elements (resonant branches + '
         f'DC edges) per seed below, full → slimmed.' if a['slim_tol'] > 0 else '- **Slimming:** off.'), '',
        '## Score', '',
        'Complex-log error 8.686·|ln(fit/target)| in dB, averaged over the positive frequencies and the unique matrix '
        'entries, then over S, Zn, Yn. Lower is better.', '',
        '| median | 5–95 % range | mean ± std | best |', '|---|---|---|---|',
        f'| {np.median(s):.3f} dB | {q[0]:.3f}–{q[1]:.3f} dB | {s.mean():.3f} ± {s.std(ddof=1) if K > 1 else 0:.3f} dB '
        f'| {s.min():.3f} dB (seed {best}) |', '',
        *([f'As the floating {n_ports}-port (fit and target converted with `circuit_model.grounded_to_floating`): '
           f'median {np.median([r["score_floating_db"] for r in runs]):.3f} dB, best seed '
           f'{next(r["score_floating_db"] for r in runs if r["seed"] == best):.3f} dB (column "floating" below).', '']
          if info['grounded'] else []),
        f'Passive (checked on a dense grid from DC up): {sum(r["passive"] for r in runs)} of {K} seeds.', '',
        '## Seeds', '',
        f'Low band {pos.min():.3g}–{10 * pos.min():.3g} Hz, high band {pos.max() / 10:.3g}–{pos.max():.3g} Hz (lowest and '
        'highest decade of the target; mean over S, Zn, Yn). S, Zn, Yn: whole band.', '',
        f'Out of band: largest curvature |d²ln H/d(ln f)²| of the self-immittances below and above the band '
        f'({a["check_fmin"]:g} Hz to {a["check_fmax"]:g} Hz, a factor {a["oob_margin"]:g} off the band; 1 = critically damped pair, a resonance of quality Q ~4 Q²), '
        'and the number of peaks above 1 there.', '',
        '| seed | score (dB) | low band | high band | S | Zn | Yn | ' + ('floating | ' if info['grounded'] else '')
        + 'curv. below | curv. above | peaks | elements | passive |',
        '|---|---|---|---|---|---|---|' + ('---|' if info['grounded'] else '') + '---|---|---|---|---|']
    for r in sorted(runs, key=lambda r: r['score_db']):
        m = r['metrics']
        mark = ' **best**' if r['seed'] == best else ''
        lines.append(f'| {r["seed"]}{mark} | {r["score_db"]:.3f} | {band(r, "lf_clog_db"):.3f} | '
                     f'{band(r, "hf_clog_db"):.3f} | {m["S"]["clog_db"]:.3f} | {m["Z"]["clog_db"]:.3f} | '
                     f'{m["Y"]["clog_db"]:.3f} | '
                     + (f'{r["score_floating_db"]:.3f} | ' if info['grounded'] else '')
                     + f'{r["wide"]["kappa_max"]["below"]:.3g} | '
                     f'{r["wide"]["kappa_max"]["above"]:.3g} | {len(r["wide"]["peaks"])} | '
                     f'{r["elements_full"]} → {r["elements"]} | {"yes" if r["passive"] else "NO"} |')
    lines += ['', '## Files', '',
              f'- `best/model.cir`: SPICE subcircuit of seed {best} (pins P1A P1B … P{n_ports}A P{n_ports}B' + (
                  ' GND; every pin is a port against GND).' if info['grounded'] else '; port p is the voltage PpA − PpB).'),
              f'- `best/model.pt`: its checkpoint (`evaluate.py`, `slim.py`, `train.py --init`).',
              '- `best/bode_{S,Z,Y}.png`: target vs fit; `best/loss.png`: training loss; `best/wide.png`: self-immittances '
              'and their curvature from the out-of-band low end to the high end, target band shaded.',
              *(['- `best/bode_floating_{S,Z,Y}.png`: the same as the floating '
                 f'{n_ports}-port (fit and target converted).'] if info['grounded'] else []),
              '- `summary.json`: everything above plus the full metrics and loss curve of every seed.']
    if a['save_all']:
        lines.append('- `seeds/model_s<seed>.{pt,cir}`: checkpoint and netlist of every seed.')
    with open(path, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def fmt_conductors(groups):
    """'{1,2} {3,4}' (1-based ports) from target_conductors(); None: one conductor per port."""
    if groups is None:
        return 'one per port (Y does not exist at the lowest frequency)'
    return ' '.join('{' + ','.join(str(p + 1) for p in g) + '}' for g in groups)


def fmt_k(v):
    v = v.detach().flatten().cpu()
    if len(v) == 1:
        return f'{v.item():.4e}'
    return f'median {v.median().item():.3e} [min {v.min().item():.3e}, max {v.max().item():.3e}]'


if __name__ == '__main__':
    main()
