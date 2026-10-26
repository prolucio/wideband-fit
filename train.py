"""Fit the trainable netlist to a target (paper, Sec. III-E). Several seeds are trained at once.

    python train.py --target data/hf.npz --seeds 100-131 --out runs/hf/100-131 [--deterministic]

Stages (Adam, all target frequencies at every step, parameters projected into their allowed range after every step):
  i)   RLC edges disconnected; the RL edges are fitted to the frequencies below 10 MHz with L_rel.
       STAGE_I_STEPS steps, learning rate STAGE_I_LR cosine-annealed to 0.
  ii)  all edges connected, whole band: L_rel + 0.1 P for the first REL_FRACTION of STAGE_II_STEPS, then L_ln + 0.1 P.
       One-cycle learning rate: LR_MAX / LR_DIV -> LR_MAX over PCT_START of the steps, then cosine to
       LR_MAX / LR_DIV / FINAL_DIV (= 2.5e-6).
  iii) pruning of every seed (prune.py).
  iv)  fine-tuning of the pruned netlists: the last FINETUNE_FRACTION of the stage ii schedule is repeated.
Adam is restarted at the start of each stage.

The seeds are independent models with their own random initialisation (torch.manual_seed(seed) before each netlist is
built), trained together by vmap over their stacked parameters; their losses are summed, so each seed receives the
gradient of its own loss. On a GPU one step is a single CUDA-graph replay. The seeds of a run share the out-of-band
frequencies of each step (drawn from the random generator after the last seed was initialised), so the result of a seed
depends on the seeds it is run with: to reproduce it, rerun the same --seeds list with --deterministic.

Output in --out: scores.csv (one row per seed), seed_<seed>.pt (checkpoint, read by evaluate.py) and seed_<seed>.cir
(SPICE subcircuit).
"""
import argparse
import csv
import os
import sys
import time
import warnings

import torch
from torch.func import functional_call, vmap

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), 'src'))

import sparams  # noqa: E402
from losses import Loss, OutOfBandPenalty
from metrics import fit_errors, is_passive, magnitude_error, phase_error
from netlist import Netlist
from prune import prune
from sparams import DEVICE, conductor_groups, load_target

# Training schedule (Sec. III-E)
STAGE_I_STEPS = 8000
STAGE_I_LR = 0.1
STAGE_I_FMAX = 1.05e7            # Hz: "frequencies below 10 MHz", with room for the grid point at 10 MHz
STAGE_II_STEPS = 30000
REL_FRACTION = 0.3               # L_rel for the first 9000 steps of stage ii, L_ln for the other 21000
LR_MAX, LR_DIV, PCT_START, FINAL_DIV = 0.02, 10, 0.1, 800
PENALTY_WEIGHT = 0.1             # weight of the out-of-band penalty P
FINETUNE_FRACTION = 1 / 3        # stage iv repeats steps 20000..29999 of stage ii


def parse_seeds(text):
    """'3' -> [3], '0-15' -> [0..15], '0,3,5' -> [0, 3, 5]"""
    seeds = []
    for part in text.split(','):
        lo, _, hi = part.partition('-')
        seeds += list(range(int(lo), int(hi) + 1)) if hi else [int(lo)]
    return seeds


class SeedBatch:
    """K netlists with the same topology and their parameters stacked into (K, ...) tensors, which are what the optimiser
    updates. The edge masks are stacked too; pruning changes them in place, which the captured steps pick up."""

    def __init__(self, seeds, target, groups):
        self.nets = []
        for seed in seeds:
            torch.manual_seed(seed)
            self.nets.append(Netlist(target.ports, target.f_min, target.f_max, groups=groups))
        self.base = self.nets[0]
        self.z0 = target.z0
        self.params = {name: torch.stack([dict(net.named_parameters())[name].detach() for net in self.nets])
                       .clone().requires_grad_() for name, _ in self.base.named_parameters()}
        self.alive = {name: torch.stack([getattr(net, name) for net in self.nets]) for name in ('rlc_alive', 'rl_alive')}

    def project(self):
        with torch.no_grad():
            self.base.project_(self.params)

    def write_back(self):
        """Copy the stacked parameters into the individual netlists."""
        with torch.no_grad():
            for k, net in enumerate(self.nets):
                for name, p in net.named_parameters():
                    p.copy_(self.params[name][k])

    def make_step(self, names, freq, loss, kind, rlc_on=True, penalty=None):
        """Function that writes into .grad of the parameters `names` the gradient, for every seed, of its loss(kind) on the
        frequencies freq (+ PENALTY_WEIGHT x P if a penalty is given), and returns the (K,) losses and penalties. On a GPU
        the function is a CUDA graph replay (the step is launch-bound), on a CPU it is the plain computation."""
        params = [self.params[n] for n in names]
        for p in params:
            if p.grad is None:      # the same .grad tensors are shared by every step function (and read by the graphs)
                p.grad = torch.zeros_like(p)

        def S_of(p, alive, f, rlc_on=True):
            return functional_call(self.base, {**p, **alive}, (f,), {'z0': self.z0, 'rlc_on': rlc_on})

        def one_seed(p, alive):
            data = loss(S_of(p, alive, freq, rlc_on), kind).mean()
            pen = penalty(S_of(p, alive, penalty.freq)) if penalty is not None else torch.zeros_like(data)
            return data, pen

        def step():
            losses, pens = vmap(one_seed)(self.params, self.alive)
            for p, g in zip(params, torch.autograd.grad(losses.sum() + PENALTY_WEIGHT * pens.sum(), params)):
                p.grad.copy_(g)
            return losses.detach(), pens.detach()

        if DEVICE != 'cuda':
            return step
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(3):   # warm-up: allocator, cuBLAS / cuSOLVER handles
                step()
        torch.cuda.current_stream().wait_stream(side)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            out = step()

        def replay():
            graph.replay()
            return out
        replay.step = step    # keeps every tensor the graph reads alive
        return replay


def stage_i(batch, target, freq, S_target):
    """RL edges only, frequencies below 10 MHz, L_rel."""
    names = ['rl_r', 'rl_l']
    low = torch.nonzero(freq <= STAGE_I_FMAX).flatten()
    loss = Loss(S_target[low], freq[low], target.z0)
    step = batch.make_step(names, freq[low], loss, 'rel', rlc_on=False)
    optimizer = torch.optim.Adam([batch.params[n] for n in names], lr=STAGE_I_LR, fused=DEVICE == 'cuda')
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, STAGE_I_STEPS)
    print(f'stage i: {len(low)} frequencies (f <= {STAGE_I_FMAX:g} Hz), {STAGE_I_STEPS} steps')
    for _ in range(STAGE_I_STEPS):
        losses, _ = step()
        optimizer.step()
        scheduler.step()
        batch.project()
    print(f'  loss {summarize(losses)}')


def run_stage_ii(batch, steps_rel, steps_ln, penalty, first_step=0):
    """Steps first_step..STAGE_II_STEPS-1 of stage ii (stage iv: first_step > 0) with a fresh Adam."""
    names = list(batch.params)
    optimizer = torch.optim.Adam([batch.params[n] for n in names], lr=LR_MAX, fused=DEVICE == 'cuda')
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=LR_MAX, total_steps=STAGE_II_STEPS,
                                                    pct_start=PCT_START, div_factor=LR_DIV, final_div_factor=FINAL_DIV)
    with warnings.catch_warnings():     # "scheduler.step() before optimizer.step()": we only fast-forward the schedule
        warnings.simplefilter('ignore')
        for _ in range(first_step):
            scheduler.step()
    switch = round(REL_FRACTION * STAGE_II_STEPS)
    log_every = max(1, STAGE_II_STEPS // 20)
    t0 = time.time()
    for i in range(first_step, STAGE_II_STEPS):
        penalty.resample()
        losses, pens = (steps_rel if i < switch else steps_ln)()
        optimizer.step()
        scheduler.step()
        batch.project()
        if i % log_every == 0 or i == STAGE_II_STEPS - 1:
            print(f'step {i:6d}  {"rel" if i < switch else "ln "} loss '
                  f'{summarize(losses)}  penalty {summarize(pens)}  lr {scheduler.get_last_lr()[0]:.2e}')
    if DEVICE == 'cuda':
        torch.cuda.synchronize()
    print(f'  {(STAGE_II_STEPS - first_step) / (time.time() - t0):.0f} it/s x {len(batch.nets)} seeds')


def summarize(v):
    v = v.detach().flatten().cpu()
    if len(v) == 1:
        return f'{v.item():.4e}'
    return f'median {v.median().item():.3e} [min {v.min().item():.3e}, max {v.max().item():.3e}]'


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--target', required=True, help='S-parameter .npz (freq, params, z0) from parse_touchstone.py')
    ap.add_argument('--seeds', default='0', help="one seed '3', a range '0-15' or a list '0,3,5'")
    ap.add_argument('--out', default='runs/train', help='output directory')
    ap.add_argument('--deterministic', action='store_true', help='bit-identical reruns (about 1.3x slower)')
    args = ap.parse_args(argv)
    if args.deterministic:
        sparams.set_deterministic()
    os.makedirs(args.out, exist_ok=True)
    seeds = parse_seeds(args.seeds)

    target = load_target(args.target)
    freq = torch.tensor(target.freq, dtype=sparams.DTYPE, device=DEVICE)
    S_target = target.S.to(DEVICE)
    groups = conductor_groups(target)
    print(f'target {args.target}: {target.ports} ports, {len(freq)} frequencies '
          f'({target.f_min:g}-{target.f_max:g} Hz{", plus DC" if (target.freq == 0).any() else ""}), '
          f'z0 {target.z0:g} Ohm; device {DEVICE}')
    print(f'galvanically connected ports: {groups if groups is not None else "none (Y does not exist at the lowest frequency)"}')
    batch = SeedBatch(seeds, target, groups)
    print(f'{len(seeds)} seeds {seeds[0]}..{seeds[-1]}; {batch.base.n_rlc} RLC and {batch.base.n_rl} RL edges per netlist')
    t_start = time.time()

    stage_i(batch, target, freq, S_target)

    loss = Loss(S_target, freq, target.z0)
    penalty = OutOfBandPenalty(target.f_min, target.f_max)
    names = list(batch.params)
    steps_rel = batch.make_step(names, freq, loss, 'rel', penalty=penalty)
    steps_ln = batch.make_step(names, freq, loss, 'ln', penalty=penalty)
    print(f'stage ii: {STAGE_II_STEPS} steps, L_rel for the first {round(REL_FRACTION * STAGE_II_STEPS)}, then L_ln')
    run_stage_ii(batch, steps_rel, steps_ln, penalty)

    print('stage iii: pruning')
    batch.write_back()
    for k, (seed, net) in enumerate(zip(seeds, batch.nets)):
        before, after = prune(net, freq, target)
        print(f'  seed {seed}: {before} -> {after} edges')
        for name in batch.alive:
            batch.alive[name][k].copy_(getattr(net, name))

    first = round((1 - FINETUNE_FRACTION) * STAGE_II_STEPS)
    print(f'stage iv: fine-tuning, steps {first}..{STAGE_II_STEPS - 1} of stage ii')
    run_stage_ii(batch, steps_rel, steps_ln, penalty, first_step=first)
    print(f'training time {time.time() - t_start:.0f} s')

    batch.write_back()
    rows = []
    for seed, net in zip(seeds, batch.nets):
        with torch.no_grad():
            errors = fit_errors(target, net(freq, z0=target.z0).cpu())
        n_rlc, n_rl, n_eps = net.edge_counts()
        row = dict(seed=seed, magnitude_db=magnitude_error(errors), phase_deg=phase_error(errors), rlc_edges=n_rlc,
                   rl_edges=n_rl, eps_edges=n_eps, passive=is_passive(net, target.z0, target.freq))
        rows.append(row)
        print(f'seed {seed}: magnitude {row["magnitude_db"]:.4f} dB, phase {row["phase_deg"]:.3f} deg, '
              f'{n_rlc} RLC + {n_rl} RL + {n_eps} R_eps edges, {"passive" if row["passive"] else "NOT PASSIVE"}')
        torch.save({'config': net.config, 'state_dict': net.state_dict(), 'seed': seed},
                   os.path.join(args.out, f'seed_{seed}.pt'))
        net.write_spice(os.path.join(args.out, f'seed_{seed}.cir'), z0=target.z0)
    with open(os.path.join(args.out, 'scores.csv'), 'w', newline='') as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


if __name__ == '__main__':
    main()
