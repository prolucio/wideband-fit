"""Remove elements from a trained model while the fit stays within a tolerance, then write the SPICE netlist.

train.py slims every seed and fine-tunes the result as part of the standard process (slim_circuit). Standalone:
    python slim.py runs/train/best/model.pt --target TARGET_S.npz [--tol 0.05] [--out runs/train/slim]
        -> <out>.pt (checkpoint with the element masks), <out>.cir, <out>_{S,Z,Y}.png

Candidates (resonant branches and DC-path edges) are tried weakest first (max |Y| over the target frequencies). A removal
is kept if the complex-log error (utils/metrics.py clog_db) of every one of S, Z/z0, Y z0 over the whole band, the low
band, the high band and at DC (if the target has a DC point and the kind exists there) rises by at most --tol dB, and the terminals, GND and every node still in use stay connected.
Private nodes left with a single neighbour are then removed with their edges (no current flows through them).
Removing elements keeps the network passive and reciprocal. Fine-tune a standalone result with
    python train.py --target TARGET_S.npz --init runs/train/slim.pt --start-step 20000 --slim-tol 0 --out runs/train_ft
"""
import argparse

import numpy as np
import torch

import circuit_model as cm
from utils.metrics import fit_metrics, one_line
from utils.plotting import plot_bode


def connected(c, alive_rlc, alive_dc):
    """True if the terminals, GND and every node with an alive edge form one connected component."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    used = set()
    for e, (a, b) in enumerate(c.edges):
        if (alive_rlc[e] if e < c.n_rlc else alive_dc[e - c.n_rlc]) and a != b:
            used.update((a, b))
            parent[find(a)] = find(b)
    root = find(c.gnd)
    return all(find(x) == root for x in used | c.required)


def remove_dangling(c, alive_rlc, alive_dc):
    """Remove the edges of private nodes whose alive edges all lead to one single neighbour (repeat until none)."""
    protected = c.required
    removed = 0
    while True:
        nbrs, edges = {}, {}
        for e, (a, b) in enumerate(c.edges):
            if (alive_rlc[e] if e < c.n_rlc else alive_dc[e - c.n_rlc]) and a != b:
                for x, y in ((a, b), (b, a)):
                    nbrs.setdefault(x, set()).add(y)
                    edges.setdefault(x, []).append(e)
        dangling = [x for x, s in nbrs.items() if x not in protected and len(s) == 1]
        if not dangling:
            return removed
        for x in dangling:
            for e in edges[x]:
                mask, i = (alive_rlc, e) if e < c.n_rlc else (alive_dc, e - c.n_rlc)
                removed += bool(mask[i])
                mask[i] = False


def slim_circuit(c, freqs, S_t, extra, z0, tol=0.05, log=print):
    """Remove elements of c in place (element masks) while every error component stays within tol dB of its start
    value (see the module docstring); returns (elements before, elements after, score before, score after)."""
    ref = lambda i: int(c.node_map[i])   # node pairs as solved (floating ports: P1B mapped to GND, the reference)
    c.edges = [(ref(int(a_)), ref(int(b_))) for a_, b_ in zip(c.pair_a.tolist(), c.pair_b.tolist())]
    c.required = {ref(t) for t in range(c.n_terminals)} | {c.gnd}

    @torch.no_grad()
    def evaluate():
        S = c(freqs, z0=z0).cpu()
        m = fit_metrics(freqs.cpu(), S_t, S, extra)
        # error components: whole band, low and high band, DC (the score leaves DC out, so it must be guarded here,
        # or the DC path is removed for free); the score is the mean of the whole-band components
        comp = [m[k][q] for k in m for q in ('clog_db', 'lf_clog_db', 'hf_clog_db', 'dc_clog_db') if q in m[k]]
        return np.array(comp), m, float(np.mean([v['clog_db'] for v in m.values()]))

    base, m0, s0 = evaluate()
    log(f'start: score {s0:.3f} dB  ({one_line(m0)})')
    with torch.no_grad():
        omega = 2 * torch.pi * freqs[freqs > 0].view(-1, 1)
        y_rlc = c.rlc_admittances(omega).abs().max(0).values.cpu()
        y_dc = c.dc_admittances(omega).abs().max(0).values.cpu()
    cands = sorted([(float(y_rlc[e]), 'rlc', e) for e in range(c.n_rlc) if c.branch_alive[e]]
                   + [(float(y_dc[e]), 'dc', e) for e in range(c.n_dc) if c.dc_alive[e]])
    n0 = len(cands)
    removed = 0
    for y, kind, e in cands:
        mask = c.branch_alive if kind == 'rlc' else c.dc_alive
        mask[e] = False
        if connected(c, c.branch_alive.tolist(), c.dc_alive.tolist()) and (evaluate()[0] - base).max() <= tol:
            removed += 1
        else:
            mask[e] = True
    alive_rlc, alive_dc = c.branch_alive.tolist(), c.dc_alive.tolist()
    n_dangling = remove_dangling(c, alive_rlc, alive_dc)
    c.branch_alive.copy_(torch.tensor(alive_rlc))
    c.dc_alive.copy_(torch.tensor(alive_dc))
    final, m1, s1 = evaluate()
    n1 = int(c.branch_alive.sum()) + int(c.dc_alive.sum())
    log(f'kept {n1} of {n0} elements ({removed} removed by the tolerance test, {n_dangling} dangling); score '
        f'{s0:.3f} -> {s1:.3f} dB (largest component rise {(final - base).max():.3f})')
    log(f'final: {one_line(m1)}')
    return n0, n1, s0, s1


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('model_file')
    ap.add_argument('--target', required=True)
    ap.add_argument('--tol', type=float, default=0.05, help='allowed rise of each error component (dB)')
    ap.add_argument('--out', default='slim')
    a = ap.parse_args()

    c, ck = cm.load_checkpoint(a.model_file)
    d = np.load(a.target)
    z0 = float(d['z0'])
    freqs = torch.tensor(d['freq'], dtype=cm.DTYPE, device=cm.DEVICE)
    S_t = torch.tensor(d['params'], dtype=cm.CDTYPE)
    extra = {k: cm.target_normalized(S_t, k, freqs.cpu(), z0) for k in 'ZY'}
    slim_circuit(c, freqs, S_t, extra, z0, a.tol)
    with torch.no_grad():
        S = c(freqs, z0=z0).cpu()
    print(c.summary())
    cm.check_passivity(c, z0=z0, extra_freqs=freqs)
    ck['state_dict'] = c.state_dict()
    torch.save(ck, a.out + '.pt')
    c.write_spice(a.out + '.cir', z0=z0)
    plot_bode(freqs.cpu(), S_t, S, f'{a.out}_S.png')
    for k, T in extra.items():
        plot_bode(freqs.cpu(), T, cm.s_to_normalized(S, k), f'{a.out}_{k}.png', kind=f'{k}n')
    print(f'wrote {a.out}.pt, {a.out}.cir, {a.out}_[SZY].png')


if __name__ == '__main__':
    main()
