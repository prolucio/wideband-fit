"""Transient behaviour of a SPICE R/L/C subcircuit, reproduced without a SPICE simulator.

    python tools/transient_check.py NETLIST.cir [NETLIST2.cir ...] [--out fig.png]

Builds the MNA descriptor system  C x' + G x = b u(t)  (x = node voltages + inductor currents) of the netlist with
port 1 driven by a 1 V step (100 ps linear rise) behind 50 Ohm and ports 2-4 terminated in 50 Ohm, then integrates it
with fixed steps:
    trapezoidal (SPICE's default method, companion-model form)   dt = 10 ps and 1 ps
    BDF2 / Gear (L-stable)                 dt = 10 ps
    BDF2 / Gear                            dt = 1 ps
    reference: BDF2 / Gear, dt = REF_H (0.5 ps) (trapezoidal is not usable as reference on ill-conditioned netlists)
and reports the step-to-step oscillation ("trapezoidal ringing") of the port voltages, plus the pole spread
(stiffness) of the system. Element values are taken verbatim from the netlist."""
import argparse
import numpy as np
import torch

Z0 = 50.0
REF_H = 0.1e-12


def build(path):
    el, pins = [], None
    for line in open(path):
        t = line.split()
        if not t or t[0].startswith('*'):
            continue
        if t[0].lower() == '.subckt':
            pins = t[2:]
        elif t[0][0] in 'RLC':
            el.append((t[0], t[1], t[2], float(t[3])))
    # floating ports: pins P1A P1B ... (port p = pin pair, P1B the reference); grounded: ... GND (every pin a port)
    ports = ([(x, pins[-1]) for x in pins[:-1]] if len(pins) % 2
             else [(pins[2 * p], pins[2 * p + 1]) for p in range(len(pins) // 2)])
    P = len(ports)
    for p, (pa, pb) in enumerate(ports):
        el.append((f'Rport{p + 1}', pa, pb, Z0))
    ref = ports[0][1]
    nodes = sorted({x for _, a, b, _ in el for x in (a, b)} - {ref})
    ix = {n: i for i, n in enumerate(nodes)}
    Ls = [e for e in el if e[0][0] == 'L']
    N = len(nodes) + len(Ls)
    G, C = np.zeros((N, N)), np.zeros((N, N))

    def st(M, a, b, v):
        for x in (a, b):
            if x != ref:
                M[ix[x], ix[x]] += v
        if a != ref and b != ref:
            M[ix[a], ix[b]] -= v
            M[ix[b], ix[a]] -= v
    k = len(nodes)
    for name, a, b, v in el:
        if name[0] == 'R':
            st(G, a, b, 1 / v)
        elif name[0] == 'C':
            st(C, a, b, v)
        else:
            if a != ref:
                G[ix[a], k] += 1
                G[k, ix[a]] -= 1
            if b != ref:
                G[ix[b], k] -= 1
                G[k, ix[b]] += 1
            C[k, k] = v
            k += 1
    b = np.zeros(N)
    b[ix[ports[0][0]]] = 1 / Z0                 # Norton equivalent of the 1 V source behind Rport1 (its - pin = ref)
    sel = lambda x, p: (x[ix[ports[p][0]]] if ports[p][0] != ref else 0) - (x[ix[ports[p][1]]] if ports[p][1] != ref else 0)
    return G, C, b, sel, P, len(el) - P


def lu_factor(A):
    return torch.linalg.lu_factor(torch.from_numpy(np.ascontiguousarray(A)))


def lu_solve(lu, b):
    return torch.linalg.lu_solve(*lu, torch.from_numpy(np.ascontiguousarray(b))[:, None]).numpy()[:, 0]


def pencil_poles(G, C, f_min=1e-2, f_max=1e12, per_shift=2.0):
    """Finite natural frequencies s of C x' + G x = 0 ((G + s C) v = 0), by shift-invert: the eigenvalues mu of
    (G + sigma C)^-1 C give s = sigma - 1/mu (mu = 0: infinite poles of a singular C, dropped). A pole is accurate
    near its shift only, so shifts sigma = 2 pi f are spaced per_shift decades over [f_min, f_max] and every pole is
    kept from the shift nearest to |s| (sigma > 0 is never a pole of a passive network). Poles with |s| outside
    [2 pi f_min, 2 pi f_max] are dropped."""
    shifts = 2 * np.pi * 10 ** np.arange(np.log10(f_min), np.log10(f_max) + per_shift, per_shift)
    keep = []
    for k, sigma in enumerate(shifts):
        mu = np.linalg.eigvals(np.linalg.solve(G + sigma * C, C))
        mu = mu[np.abs(mu) > 1e-14 * np.abs(mu).max()]
        s = sigma - 1 / mu
        near = np.abs(np.log10(np.abs(s) + 1e-300) - np.log10(sigma)) <= per_shift / 2
        if k == 0:
            near |= np.abs(s) < sigma
        if k == len(shifts) - 1:
            near |= np.abs(s) > sigma
        keep.append(s[near])
    s = np.concatenate(keep)
    return s[(np.abs(s) >= 2 * np.pi * f_min) & (np.abs(s) <= 2 * np.pi * f_max)]


def u(t, rise=100e-12):
    return np.clip(t / rise, 0, 1)


def simulate(G, C, b, h, T, method):
    n = int(round(T / h))
    t = np.arange(n + 1) * h
    x = np.zeros((n + 1, len(b)))
    if method == 'trap':
        # SPICE-style trapezoidal: companion models for the reactive elements only. The derivative history
        # xd (capacitor currents / inductor voltages divided by C, L) is carried; KCL and resistors are enforced
        # exactly at t_{n+1}:  (2C/h + G) x1 = b u1 + C (2 x0 / h + xd0),  xd1 = 2 (x1 - x0) / h - xd0
        lu = lu_factor(2 * C / h + G)
        xd = np.zeros(len(b))
        for i in range(n):
            x[i + 1] = lu_solve(lu, b * u(t[i + 1]) + C @ (2 * x[i] / h + xd))
            xd = 2 * (x[i + 1] - x[i]) / h - xd
    else:                                         # BDF2 (Gear 2), first step backward Euler
        lu1 = lu_factor(C / h + G)
        x[1] = lu_solve(lu1, C @ x[0] / h + b * u(t[1]))
        lu = lu_factor(1.5 * C / h + G)
        for i in range(1, n):
            x[i + 1] = lu_solve(lu, C @ (2 * x[i] - 0.5 * x[i - 1]) / h + b * u(t[i + 1]))
    return t, x


def ringing(v):
    """step-to-step alternating component: mean |second difference| relative to the signal range"""
    d2 = v[2:] - 2 * v[1:-1] + v[:-2]
    return np.abs(d2).max() / (np.abs(v).max() + 1e-30)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('netlists', nargs='+')
    ap.add_argument('--out', default=None, help='.npz with the traces')
    ap.add_argument('--T', type=float, default=4e-9)
    a = ap.parse_args()
    traces = {}
    for r, path in enumerate(a.netlists):
        G, C, b, sel, P, n_el = build(path)
        w = pencil_poles(G, C)
        w = w[np.isfinite(w) & (np.abs(w) < 1e30) & (np.abs(w) > 0)]
        print(f'== {path}: {n_el} elements, {len(b)} states; poles |s| from {np.abs(w).min():.2e} to {np.abs(w).max():.2e} '
              f'rad/s (stiffness {np.abs(w).max() / np.abs(w).min():.1e}); condition number of G {np.linalg.cond(G):.1e}')
        ref_t, ref_x = simulate(G, C, b, REF_H, a.T, 'gear')
        ref = {p: np.array([sel(x, p) for x in ref_x]) for p in (0, 1)}
        for h, method in ((10e-12, 'trap'), (1e-12, 'trap'), (10e-12, 'gear'), (1e-12, 'gear')):
            try:
                with np.errstate(all='ignore'):
                    t, x = simulate(G, C, b, h, a.T, method)
            except ValueError:                      # NaN reached the solver
                print(f'   {method:4s} dt {h * 1e12:5.1f} ps: BLOW-UP (non-finite solution)')
                continue
            for p in (0, 1):
                v = np.array([sel(xx, p) for xx in x])
                vr = np.interp(t, ref_t, ref[p])
                err = np.abs(v - vr).max()
                bad = (not np.all(np.isfinite(v))) or np.abs(v).max() > 10
                print(f'   {method:4s} dt {h * 1e12:5.1f} ps  port {p + 1}: ' + ('BLOW-UP' if bad else
                      f'max |error| vs reference {err:.2e} V, step-to-step ringing {ringing(v):.2e}'))
                traces[f'{r}|{method}|{h:g}|{p}'] = np.stack([t, v])
        for p in (0, 1):
            traces[f'{r}|ref|{REF_H:g}|{p}'] = np.stack([ref_t, ref[p]])
    if a.out:
        np.savez(a.out, names=np.array(a.netlists), **traces)
        print('traces', a.out)


if __name__ == '__main__':
    main()
