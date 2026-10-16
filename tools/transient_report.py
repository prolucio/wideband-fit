"""Transient behaviour of a fitted SPICE netlist, without a SPICE simulator:

    python tools/transient_report.py NETLIST.cir --band F_LO F_HI [--out report]

1. Poles. The natural frequencies s of the MNA system C x' + G x = 0 (ports terminated in 50 Ohm), with
   f = |Im s| / 2 pi and Q = |s| / (2 |Re s|). Every pole of a passive network has Re s <= 0; a pole pair with Q > 1/2
   rings. Listed: the ringing poles below and above the band [F_LO, F_HI] (the target does not ask for them), and
   the slowest decay time.
2. Step responses. Port 1 driven by a 1 V step (100 ps rise) behind 50 Ohm, the other ports in 50 Ohm, on two time
   scales: fast (5 ns, dt 1 ps) and slow (200 us, dt 10 ns), BDF2/Gear (L-stable). The fast one is repeated with
   SPICE-style trapezoidal integration (dt 10 ps and 1 ps): the difference shows trapezoidal ringing.
Outputs: report.json and report.png (port voltages on both time scales).
"""
import argparse
import json
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

from transient_check import build, pencil_poles, simulate


def poles(G, C):
    w = pencil_poles(G, C)
    w = w[np.isfinite(w) & (np.abs(w) > 0)]
    f = np.abs(w.imag) / (2 * math.pi)
    Q = np.abs(w) / (2 * np.maximum(-w.real, 1e-9 * np.abs(w)))   # Re s within round-off of 0: Q capped at 5e8
    return w, f, Q


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('netlist')
    ap.add_argument('--band', type=float, nargs=2, required=True, metavar=('F_LO', 'F_HI'), help='target band (Hz)')
    ap.add_argument('--out', default=None, help='output stem (default: the netlist path without .cir + _transient)')
    a = ap.parse_args()
    out = a.out or a.netlist.rsplit('.', 1)[0] + '_transient'
    f_lo, f_hi = a.band
    G, C, b, sel, P, n_el = build(a.netlist)

    w, f, Q = poles(G, C)
    ring = Q > 0.5
    unstable = int((w.real > 1e-6 * np.abs(w)).sum())   # beyond round-off
    below = sorted(((float(fi), float(qi)) for fi, qi in zip(f[ring], Q[ring]) if fi < f_lo), key=lambda x: -x[1])
    above = sorted(((float(fi), float(qi)) for fi, qi in zip(f[ring], Q[ring]) if fi > f_hi), key=lambda x: -x[1])
    tau = float(1 / np.min(-w.real[w.real < 0])) if (w.real < 0).any() else float('inf')
    rep = {'netlist': a.netlist, 'elements': n_el, 'states': len(b), 'band': [f_lo, f_hi], 'n_poles': len(w),
           'unstable_poles': unstable, 'slowest_decay_s': tau,
           'ringing_below_band': below, 'ringing_above_band': above,
           'max_Q_in_band': float(Q[ring & (f >= f_lo) & (f <= f_hi)].max()) if (ring & (f >= f_lo) & (f <= f_hi)).any() else 0.0}

    sims = {}
    for name, T, h, meth in (('fast_gear', 5e-9, 1e-12, 'gear'), ('fast_trap_10ps', 5e-9, 10e-12, 'trap'),
                             ('fast_trap_1ps', 5e-9, 1e-12, 'trap'), ('slow_gear', 200e-6, 10e-9, 'gear')):
        with np.errstate(all='ignore'):
            try:
                t, x = simulate(G, C, b, h, T, meth)
                v = np.array([[sel(xx, p) for p in range(P)] for xx in x])
            except ValueError:
                t, v = None, None
        ok = v is not None and np.all(np.isfinite(v)) and np.abs(v).max() < 10
        sims[name] = (t, v)
        rep[name] = {'ok': bool(ok), 'final_V': v[-1].tolist() if ok else None,
                     'max_abs_V': float(np.abs(v).max()) if ok else None}
    tg, vg = sims['fast_gear']
    for name in ('fast_trap_10ps', 'fast_trap_1ps'):
        t, v = sims[name]
        if rep[name]['ok'] and rep['fast_gear']['ok']:
            rep[name]['max_diff_vs_gear_V'] = float(np.max(np.abs(v - np.array([np.interp(t, tg, vg[:, p])
                                                                                     for p in range(P)]).T)))
    json.dump(rep, open(out + '.json', 'w'), indent=1)

    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for k, (name, unit, s) in enumerate((('fast_gear', 'ns', 1e9), ('slow_gear', 'us', 1e6))):
        t, v = sims[name]
        if v is not None:
            for p in range(P):
                ax[k].plot(t * s, v[:, p], lw=1, label=f'V{p + 1}')
        ax[k].set_xlabel(f'time ({unit})'); ax[k].set_ylabel('port voltage (V)'); ax[k].grid(alpha=0.3)
        ax[k].set_title(f'{name.split("_")[0]} step response (Gear)')
    ax[1].set_xscale('symlog', linthresh=1e-3)
    ax[0].legend()
    fig.tight_layout(); fig.savefig(out + '.png', dpi=130); plt.close(fig)

    print(f'{a.netlist}: {n_el} elements, {len(w)} poles, unstable {unstable}, slowest decay {tau:.3g} s')
    print(f'  ringing poles (Q > 0.5) below the band: {len(below)}' +
          (' (' + ', '.join(f'{fi:.3g} Hz Q {qi:.2g}' for fi, qi in below[:5]) + ')' if below else ''))
    print(f'  ringing poles above the band: {len(above)}' +
          (' (' + ', '.join(f'{fi:.3g} Hz Q {qi:.2g}' for fi, qi in above[:5]) + ')' if above else ''))
    for name in ('fast_gear', 'fast_trap_10ps', 'fast_trap_1ps', 'slow_gear'):
        r = rep[name]
        extra = f", trapezoidal vs Gear max diff {r['max_diff_vs_gear_V']:.2e} V" if 'max_diff_vs_gear_V' in r else ''
        print(f'  {name}: ' + (f"ok, final V {np.round(r['final_V'], 4).tolist()}, max |V| {r['max_abs_V']:.3g}{extra}"
                               if r['ok'] else 'BLOW-UP'))
    print(f'  -> {out}.json, {out}.png')


if __name__ == '__main__':
    main()
