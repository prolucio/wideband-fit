"""Figures of a training run as PNG plus CSV files holding exactly the plotted points (for pgfplots).

    python tools/export_figures.py runs/model-hf [--seed best|median|N]

Reads RUN/summary.json (target, settings, seeds) and the model of the chosen seed: best (RUN/best/model.{pt,cir}),
median (the lower median of the seed scores, from RUN/seeds/, needs --save-all) or a seed number. Writes
RUN/figures_<seed choice>/ (e.g. figures_best/, figures_median/), with seed.txt naming the seed and its score:

  bode_{S,Z,Y}.csv/.png   target vs fit in the target band, every entry ij with i <= j (the matrices are symmetric):
                          f_Hz, t<ij>_dB, t<ij>_deg, f<ij>_dB, f<ij>_deg (t = target, f = fit; the fit phase is
                          unwrapped next to the target's, as in the PNG). Z, Y normalized (Z/z0, Y z0). A missing
                          target (Z at DC of isolated conductors) is nan, and so is the fit at DC there. A DC row has f_Hz = 0 (not on a log axis).
  wide_fit.csv            the fit from --fmin to --fmax (default 1 Hz to 10 GHz, 200 points per decade): f_Hz,
                          Z<ii>_abs, Y<ii>_abs (normalized self-immittances), curv (|d2 ln H/d(ln f)2|, max over them;
                          nan at the two ends), band (1 inside the target band, 0 outside)
  wide_target.csv         the target self-immittances: f_Hz, Z<ii>_abs, Y<ii>_abs (positive frequencies)
  wide.png                both of the above
  loss.csv/.png           training loss of the seed (100 points): step, loss, oob (out-of-band penalty, if used)
  seeds.csv               one row per seed: seed, score_dB, low_dB, high_dB, S_dB, Z_dB, Y_dB, curv_below, curv_above,
                          peaks, passive
  transient_fast.csv      step response, port 1 driven (1 V step, 100 ps rise, behind 50 Ohm), other ports in 50 Ohm,
                          BDF2/Gear dt 1 ps, written every 5 ps: t_ns, V1 .. Vn
  transient_slow.csv      the same up to 200 us, dt 10 ns, written on a log time grid (~1000 points): t_us, V1 .. Vn
  transient.png           both
  poles.csv/.png          natural frequencies of the netlist (ports in 50 Ohm): f_Hz, Q, re_s, im_s (upper half plane
                          only), with the target band in the PNG
"""
import argparse
import csv
import json
import math
import os
import sys

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, HERE)
import matplotlib  # noqa: E402
matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

import circuit_model as cm  # noqa: E402
from utils.plotting import plot_bode, plot_loss  # noqa: E402
from utils.wideband import plot_wide, wide_sweep  # noqa: E402
from transient_check import build, pencil_poles, simulate  # noqa: E402


def write_csv(path, header, cols):
    rows = np.column_stack([np.asarray(c, dtype=float) for c in cols])
    with open(path, 'w', newline='') as fh:
        w = csv.writer(fh)
        w.writerow(header)
        for r in rows:
            w.writerow(['nan' if not np.isfinite(x) else f'{x:.9g}' for x in r])


def bode_csv(path, freq, T, P, kind):
    n = T.shape[-1]
    header, cols = ['f_Hz'], [freq]
    db = lambda x: 20 * np.log10(np.abs(x) + 1e-300)
    for i in range(n):
        for j in range(i, n):
            t, p = T[:, i, j], P[:, i, j]
            missing = ~np.isfinite(t)   # target does not exist there: nan, fit phase not unwrapped
            t_ph = np.angle(t, deg=True)
            p_ph = np.where(missing, np.angle(p, deg=True), t_ph + np.angle(p * np.conj(t), deg=True))
            ij = f'{i + 1}{j + 1}'
            header += [f't{ij}_dB', f't{ij}_deg', f'f{ij}_dB', f'f{ij}_deg']
            dc_missing = missing & (freq == 0)   # fit at DC without a target: leakage-defined, rounding noise
            cols += [np.where(missing, np.nan, db(t)), t_ph, np.where(dc_missing, np.nan, db(p)),
                     np.where(dc_missing, np.nan, p_ph)]
    write_csv(path, header, cols)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('run', help='training output directory (summary.json, best/)')
    ap.add_argument('--seed', default='best', help="'best', 'median' (lower median of the scores) or a seed number")
    ap.add_argument('--fmin', type=float, default=1.0, help='wide plot: low end (Hz)')
    ap.add_argument('--fmax', type=float, default=1e10, help='wide plot: high end (Hz)')
    a = ap.parse_args()
    info = json.load(open(os.path.join(a.run, 'summary.json')))
    args = info['args']
    ranked = sorted(info['runs'], key=lambda r: r['score_db'])
    if a.seed == 'best':
        run, stem = ranked[0], os.path.join(a.run, 'best', 'model')
    else:
        run = ranked[(len(ranked) - 1) // 2] if a.seed == 'median' else next(r for r in ranked
                                                                           if r['seed'] == int(a.seed))
        stem = os.path.join(a.run, 'seeds', f'model_s{run["seed"]}')
    out = os.path.join(a.run, f'figures_{a.seed}')
    os.makedirs(out, exist_ok=True)
    with open(os.path.join(out, 'seed.txt'), 'w') as fh:
        fh.write(f'seed {run["seed"]} ({a.seed}): score {run["score_db"]:.4f} dB, rank '
                 f'{ranked.index(run) + 1} of {len(ranked)}; median of the run {np.median([r["score_db"] for r in ranked]):.4f} dB\n')
    target = args['target']
    if not os.path.isabs(target) and not os.path.exists(target):
        target = os.path.join(os.path.dirname(HERE), target)
    d = np.load(target)
    z0 = float(d['z0'])
    freq = d['freq']
    pos = freq[freq > 0]
    f_lo, f_hi = pos.min(), pos.max()
    S_t = torch.tensor(d['params'], dtype=cm.CDTYPE)
    c, _ = cm.load_checkpoint(stem + '.pt')
    with torch.no_grad():
        S = c(torch.tensor(freq, dtype=cm.DTYPE, device=cm.DEVICE), z0=z0).cpu()

    # 1. Bode plots in the band
    fr = torch.tensor(freq, dtype=cm.DTYPE)
    for kind in 'SZY':
        T = S_t if kind == 'S' else cm.target_normalized(S_t, kind, freq, z0)
        P = S if kind == 'S' else cm.s_to_normalized(S, kind)
        bode_csv(os.path.join(out, f'bode_{kind}.csv'), freq, T.numpy(), P.numpy(), kind)
        plot_bode(fr, T, P, os.path.join(out, f'bode_{kind}.png'), kind=kind if kind == 'S' else f'{kind}n')

    # 2. wide band
    margin = args.get('oob_margin', 3.0)
    w = wide_sweep(c, z0, f_lo, f_hi, a.fmin, a.fmax, margin=margin)
    n = w['H']['Z'].shape[1]
    curv = np.concatenate([[np.nan], w['kappa'], [np.nan]])
    band = ((w['f'] >= f_lo) & (w['f'] <= f_hi)).astype(float)
    write_csv(os.path.join(out, 'wide_fit.csv'),
              ['f_Hz'] + [f'Z{i + 1}{i + 1}_abs' for i in range(n)] + [f'Y{i + 1}{i + 1}_abs' for i in range(n)]
              + ['curv', 'band'],
              [w['f']] + [np.abs(w['H']['Z'][:, i]) for i in range(n)] + [np.abs(w['H']['Y'][:, i]) for i in range(n)]
              + [curv, band])
    H_t = {k: torch.diagonal(cm.target_normalized(S_t, k, freq, z0), dim1=-2, dim2=-1).numpy() for k in 'ZY'}
    k_pos = freq > 0
    write_csv(os.path.join(out, 'wide_target.csv'),
              ['f_Hz'] + [f'Z{i + 1}{i + 1}_abs' for i in range(n)] + [f'Y{i + 1}{i + 1}_abs' for i in range(n)],
              [freq[k_pos]] + [np.abs(H_t['Z'][k_pos, i]) for i in range(n)]
              + [np.abs(H_t['Y'][k_pos, i]) for i in range(n)])
    plot_wide(w, os.path.join(out, 'wide.png'), freq, H_t, f_lo, f_hi, f'seed {run["seed"]} ({a.seed})')

    # 3. loss and seeds
    loss = np.array(run['loss_curve'])
    steps = np.linspace(args.get('start_step', 0), args['steps'] - 1, len(loss)).round()
    oob = np.array(run['oob_curve']) if run.get('oob_curve') else np.full(len(loss), np.nan)
    write_csv(os.path.join(out, 'loss.csv'), ['step', 'loss', 'oob'], [steps, loss, oob])
    switch = round(args['switch_frac'] * args['steps']) - args.get('start_step', 0)
    plot_loss(loss.tolist(), os.path.join(out, 'loss.png'), switch=switch * len(loss) / max(1, args['steps']))
    band_db = lambda r, q: float(np.mean([v[q] for v in r['metrics'].values()]))
    rs = sorted(info['runs'], key=lambda r: r['score_db'])
    write_csv(os.path.join(out, 'seeds.csv'),
              ['seed', 'score_dB', 'low_dB', 'high_dB', 'S_dB', 'Z_dB', 'Y_dB', 'curv_below', 'curv_above', 'peaks',
               'passive'],
              [[r['seed'] for r in rs], [r['score_db'] for r in rs], [band_db(r, 'lf_clog_db') for r in rs],
               [band_db(r, 'hf_clog_db') for r in rs]] + [[r['metrics'][k]['clog_db'] for r in rs] for k in 'SZY']
              + [[r['wide']['kappa_max']['below'] for r in rs], [r['wide']['kappa_max']['above'] for r in rs],
                 [len(r['wide']['peaks']) for r in rs], [float(r['passive']) for r in rs]])

    # 4. transient and poles of the netlist
    G, C, b, sel, P, _ = build(stem + '.cir')
    tf, xf = simulate(G, C, b, 1e-12, 5e-9, 'gear')
    vf = np.array([[sel(x, p) for p in range(P)] for x in xf])[::5]
    write_csv(os.path.join(out, 'transient_fast.csv'), ['t_ns'] + [f'V{p + 1}' for p in range(P)],
              [tf[::5] * 1e9] + [vf[:, p] for p in range(P)])
    ts, xs = simulate(G, C, b, 10e-9, 200e-6, 'gear')
    vs = np.array([[sel(x, p) for p in range(P)] for x in xs])
    idx = np.unique(np.concatenate([[0], np.round(np.logspace(0, math.log10(len(ts) - 1), 1000)).astype(int)]))
    write_csv(os.path.join(out, 'transient_slow.csv'), ['t_us'] + [f'V{p + 1}' for p in range(P)],
              [ts[idx] * 1e6] + [vs[idx, p] for p in range(P)])
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for p in range(P):
        ax[0].plot(tf[::5] * 1e9, vf[:, p], lw=1, label=f'V{p + 1}')
        ax[1].plot(ts[idx] * 1e6, vs[idx, p], lw=1, label=f'V{p + 1}')
    ax[0].set_xlabel('time (ns)'); ax[1].set_xlabel('time (µs)'); ax[1].set_xscale('log')
    for x in ax:
        x.set_ylabel('port voltage (V)'); x.grid(alpha=0.3)
    ax[0].legend(); ax[0].set_title('step response, port 1 driven (Gear)')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'transient.png'), dpi=130); plt.close(fig)

    s = pencil_poles(G, C)
    s = s[s.imag >= 0]
    fp = np.abs(s.imag) / (2 * math.pi)
    Q = np.abs(s) / (2 * np.maximum(-s.real, 1e-9 * np.abs(s)))
    write_csv(os.path.join(out, 'poles.csv'), ['f_Hz', 'Q', 're_s', 'im_s'], [fp, Q, s.real, s.imag])
    fig, ax = plt.subplots(figsize=(7, 4))
    ring = (Q > 0.5) & (fp > 0)
    ax.loglog(fp[ring], Q[ring], '.', ms=4)
    ax.axvspan(f_lo, f_hi, color='0.9', zorder=0)
    ax.axhline(0.5, color='r', ls='--', lw=1, label='Q = 1/2 (no ringing below)')
    ax.set_xlabel('pole frequency (Hz)'); ax.set_ylabel('Q'); ax.grid(True, which='both', alpha=0.3); ax.legend()
    ax.set_title('ringing poles of the netlist (grey: target band)')
    fig.tight_layout(); fig.savefig(os.path.join(out, 'poles.png'), dpi=130); plt.close(fig)
    print(f'figures and csv in {out}: ' + ', '.join(sorted(os.listdir(out))))


if __name__ == '__main__':
    main()
