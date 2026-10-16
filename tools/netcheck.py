"""Transient-friendliness of SPICE R/L/C netlists:
    python tools/netcheck.py OUT.json NETLIST.cir [...]
Per netlist: element value ranges (leaks excluded), cond(G), and stability of a 0.5 ns step response (port 1 driven,
1 V / 100 ps behind 50 Ohm, other ports 50 Ohm) with SPICE-style trapezoidal (10, 1, 0.1 ps) and Gear/BDF2 (1, 0.1 ps)
fixed steps. 'ok' = finite and |V| < 10 V everywhere."""
import json, os, sys
import numpy as np
from transient_check import build, simulate


def check(path):
    vals = {'R': [], 'L': [], 'C': []}
    for line in open(path):
        t = line.split()
        if t and t[0][0] in 'RLC' and not t[0].startswith('Rleak'):
            vals[t[0][0]].append(float(t[3]))
    G, C, b, sel, P, n = build(path)
    out = {'n_elements': n, 'cond_G': float(np.linalg.cond(G)),
           'range': {k: [min(v), max(v)] for k, v in vals.items() if v}}
    for h, meth in ((10e-12, 'trap'), (1e-12, 'trap'), (0.1e-12, 'trap'), (1e-12, 'gear'), (0.1e-12, 'gear')):
        try:
            with np.errstate(all='ignore'):
                t, x = simulate(G, C, b, h, 0.5e-9, meth)
            v = np.array([sel(xx, 1) for xx in x])
            ok = bool(np.all(np.isfinite(v)) and np.abs(v).max() < 10)
        except ValueError:
            ok = False
        out[f'{meth}_{h * 1e12:g}ps'] = ok
    return out


if __name__ == '__main__':
    res = {p: check(p) for p in sys.argv[2:]}
    for p, r in res.items():
        print(os.path.basename(p), f"cond {r['cond_G']:.1e}", ' '.join(f"{k}={'ok' if v else 'FAIL'}" for k, v in r.items() if k.startswith(('trap', 'gear'))))
    json.dump(res, open(sys.argv[1], 'w'), indent=1)
