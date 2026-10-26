"""Pruning (stage iii, Sec. III-E): remove the edges that the fit does not need.

Edges (RLC and RL) are tried one by one, weakest first (smallest maximum admittance over the target frequencies). Removing
an edge sets its admittance to 0. The removal is kept if
  - the terminals, the reference node and every node that still has an edge stay connected, and
  - none of the logarithmic errors rises by more than PRUNE_TOL (in neper, 0.005 = 0.043 dB) above its value before
    pruning. The errors are |ln(h_hat / h)| for each of S, Z, Y, averaged over the whole band, its lowest decade, its
    highest decade, and at DC if the target has a DC point (the DC point is not part of the reported errors: without
    this guard pruning would remove the DC path for free).
Finally, internal nodes left with a single neighbour are removed with their edges (no current flows through them).
Removing edges keeps the netlist passive and reciprocal.
"""
import numpy as np
import torch

from metrics import fit_errors

PRUNE_TOL = 0.005
COMPONENTS = ('ln', 'ln_low', 'ln_high', 'ln_dc')


def _connected(edges, alive, required, reference):
    """True if the nodes in `required` and every node with a live edge form one connected component."""
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    used = set()
    for (a, b), ok in zip(edges, alive):
        if ok and a != b:
            used.update((a, b))
            parent[find(a)] = find(b)
    root = find(reference)
    return all(find(x) == root for x in used | required)


def _remove_dangling(edges, alive, required):
    """Remove the edges of non-required nodes whose live edges all lead to one single neighbour (repeat until none)."""
    removed = 0
    while True:
        neighbours, incident = {}, {}
        for e, ((a, b), ok) in enumerate(zip(edges, alive)):
            if ok and a != b:
                for x, y in ((a, b), (b, a)):
                    neighbours.setdefault(x, set()).add(y)
                    incident.setdefault(x, []).append(e)
        dangling = [x for x, s in neighbours.items() if x not in required and len(s) == 1]
        if not dangling:
            return removed
        for x in dangling:
            for e in incident[x]:
                removed += alive[e]
                alive[e] = False


def prune(net, freq, target, tol=PRUNE_TOL):
    """Prune `net` in place (its rlc_alive and rl_alive masks). freq: the target frequencies on the compute device,
    target: the Target. Returns (edges before, edges after)."""
    edges = list(zip(net.pair_a.tolist(), net.pair_b.tolist()))
    required = set(range(net.n_terminals))
    n_rlc = net.n_rlc
    alive = net.rlc_alive.tolist() + net.rl_alive.tolist()

    def set_alive(e, value):
        (net.rlc_alive if e < n_rlc else net.rl_alive)[e if e < n_rlc else e - n_rlc] = value

    @torch.no_grad()
    def errors():
        errors = fit_errors(target, net(freq, z0=target.z0).cpu())
        return np.array([e[c] for e in errors.values() for c in COMPONENTS if c in e])

    reference = errors()
    with torch.no_grad():
        omega = 2 * torch.pi * freq[freq > 0].view(-1, 1)
        strength = torch.cat([net.rlc_admittances(omega).abs().max(0).values,
                              net.rl_admittances(omega).abs().max(0).values]).cpu()
    candidates = sorted((float(strength[e]), e) for e in range(len(alive)) if alive[e])
    n_before = len(candidates)
    for _, e in candidates:
        alive[e] = False
        set_alive(e, False)
        if not (_connected(edges, alive, required, net.reference) and (errors() - reference).max() <= tol):
            alive[e] = True
            set_alive(e, True)
    _remove_dangling(edges, alive, required)
    net.rlc_alive.copy_(torch.tensor(alive[:n_rlc]))
    net.rl_alive.copy_(torch.tensor(alive[n_rlc:]))
    return n_before, sum(alive)
