"""Score saved models against a target.

    python evaluate.py --target TARGET_S.npz MODEL.pt [MODEL.pt ...] [--plots] [--passivity] [--wide]

score = mean over S, Z/z0, Y z0 of the complex-log error 8.686 |ln(fit / target)| (dB), averaged over the positive
frequencies and the unique matrix entries (utils/metrics.py). Also printed: the low-band (lowest decade of the target)
and high-band (highest decade) values and the number of elements left. --plots writes <model>_{S,Z,Y}.png next to each model.
--wide checks the behaviour far outside the band (DC, --fmin to --fmax; utils/wideband.py) and writes <model>_wide.png.
--floating: a model with grounded ports (2n ports) is scored as the floating n-port (circuit_model.grounded_to_floating)
against the target, which is then a floating n-port (e.g. the solver's own floating export, on its own grid);
--plots then writes <model>_floating_{S,Z,Y}.png.
"""
import argparse
import os

import numpy as np
import torch

import circuit_model as cm
from utils.metrics import fit_metrics, score
from utils.plotting import plot_bode
from utils.wideband import describe, plot_wide, wide_sweep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('models', nargs='+')
    ap.add_argument('--target', required=True)
    ap.add_argument('--plots', action='store_true')
    ap.add_argument('--passivity', action='store_true', help='also check passivity on a dense grid')
    ap.add_argument('--wide', action='store_true', help='also check the behaviour outside the target band')
    ap.add_argument('--fmin', type=float, default=1.0, help='--wide: low end (Hz)')
    ap.add_argument('--fmax', type=float, default=1e10, help='--wide: high end (Hz)')
    ap.add_argument('--floating', action='store_true',
                    help='grounded models: score the floating n-port against a floating n-port target')
    a = ap.parse_args()
    d = np.load(a.target)
    z0 = float(d['z0'])
    freqs = torch.tensor(d['freq'], dtype=cm.DTYPE, device=cm.DEVICE)
    S_t = torch.tensor(d['params'], dtype=cm.CDTYPE)
    extra = {k: cm.target_normalized(S_t, k, d['freq'], z0) for k in 'ZY'}
    scores = []
    for path in a.models:
        c, _ = cm.load_checkpoint(path)
        with torch.no_grad():
            S = c(freqs, z0=z0).cpu()
        if a.floating:
            if not c.grounded:
                raise SystemExit(f'{path}: --floating needs a model with grounded ports')
            S = cm.grounded_to_floating(S, z0)
        m = fit_metrics(freqs.cpu(), S_t, S, extra)
        band = lambda q: np.mean([v[q] for v in m.values()])
        n_el = int(c.branch_alive.sum()) + int(c.dc_alive.sum())
        scores.append(score(m))
        print(f'{path}: score {score(m):.3f} dB  (low band {band("lf_clog_db"):.3f}, high band {band("hf_clog_db"):.3f}; '
              f'S {m["S"]["clog_db"]:.3f}, Z {m["Z"]["clog_db"]:.3f}, Y {m["Y"]["clog_db"]:.3f})  {n_el} elements')
        if a.passivity:
            cm.check_passivity(c, z0=z0, extra_freqs=freqs)
        if a.wide:
            pos = d['freq'][d['freq'] > 0]
            w = wide_sweep(c, z0, pos.min(), pos.max(), a.fmin, a.fmax)
            print(f'   wide band {a.fmin:g}-{a.fmax:g} Hz: {describe(w)}')
            H_t = {k: torch.diagonal(H, dim1=-2, dim2=-1).numpy() for k, H in extra.items()}
            plot_wide(w, os.path.splitext(path)[0] + '_wide.png', d['freq'], H_t, pos.min(), pos.max(),
                      os.path.basename(path))
        if a.plots:
            root = os.path.splitext(path)[0] + ('_floating' if a.floating else '')
            plot_bode(freqs.cpu(), S_t, S, f'{root}_S.png')
            for k, T in extra.items():
                plot_bode(freqs.cpu(), T, cm.s_to_normalized(S, k), f'{root}_{k}.png', kind=f'{k}n')
    if len(scores) > 1:
        s = np.array(scores)
        print(f'{len(s)} models: mean {s.mean():.3f}  std {s.std(ddof=1):.3f}  median {np.median(s):.3f}  best {s.min():.3f}')


if __name__ == '__main__':
    main()
