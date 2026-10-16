"""Behaviour of a fitted circuit far outside the target band: DC and a dense log grid (default 1 Hz to 10 GHz).

wide_sweep evaluates the self-immittances diag Z/z0 and diag Y z0 on the grid and their curvature
kappa = |d2 ln H / d(ln f)2| (second differences on the uniform log grid), the same measure as the out-of-band penalty
of train.py: a real pole or zero bends ln H by at most 1/2, a critically damped pair by 1, a resonance of quality Q by
~4 Q^2. Reported per region (below the band, inside it, above it; a factor 'margin' on each side of the band belongs to
none: room for resonances just outside the band whose tails reach into it): the largest curvature and the out-of-band peaks
above KAPPA_MAX (resonances the target does not ask for). plot_wide draws |diag Z/z0|, |diag Y z0| and kappa with the
target overlaid and the band shaded.
"""
import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

import circuit_model as cm

KAPPA_MAX = 1.0


@torch.no_grad()
def wide_sweep(c, z0, f_lo, f_hi, fmin=1.0, fmax=1e10, per_decade=200, margin=3.0):
    n = int(round(math.log10(fmax / fmin) * per_decade)) + 1
    f = np.logspace(math.log10(fmin), math.log10(fmax), n)
    S = torch.cat([c(torch.tensor(ch, dtype=cm.DTYPE, device=cm.DEVICE), z0=z0).cpu()
                   for ch in np.array_split(f, max(1, n // 1000))])
    S_dc = c(torch.zeros(1, dtype=cm.DTYPE, device=cm.DEVICE), z0=z0).cpu()[0]
    H = {k: torch.diagonal(cm.s_to_normalized(S, k), dim1=-2, dim2=-1).numpy() for k in 'ZY'}
    h = math.log(10) / per_decade
    kappa = {k: np.abs(np.diff(np.log(v), 2, axis=0)) / h ** 2 for k, v in H.items()}   # at f[1:-1]
    kap = np.maximum(kappa['Z'], kappa['Y']).max(1)
    fk = f[1:-1]
    lo, hi = f_lo / margin, f_hi * margin   # margin: room for resonances just outside the band
    regions = {'below': fk < lo, 'band': (fk >= f_lo) & (fk <= f_hi), 'above': fk > hi}
    out = {'f': f, 'S': S.numpy(), 'H': H, 'kappa': kap, 'S_dc': S_dc.numpy(),
           'kappa_max': {r: float(kap[m].max()) if m.any() else float('nan') for r, m in regions.items()}}
    # out-of-band peaks: local maxima of kappa above KAPPA_MAX outside the band
    oob = regions['below'] | regions['above']
    peaks = [i for i in range(1, len(kap) - 1)
             if oob[i] and kap[i] > KAPPA_MAX and kap[i] >= kap[i - 1] and kap[i] >= kap[i + 1]]
    out['peaks'] = [(float(fk[i]), float(kap[i])) for i in peaks]
    # DC: Y z0 (exists: leak), Z/z0 only if I - S is regular
    eye = np.eye(S_dc.shape[-1])
    Sd = S_dc.numpy()
    out['Y_dc'] = np.diag(np.linalg.solve(eye + Sd, eye - Sd))
    out['Z_dc_exists'] = bool(np.linalg.svd(eye - Sd, compute_uv=False)[-1] > 1e-12)
    return out


def describe(w):
    k = w['kappa_max']
    s = (f"curvature max: below band {k['below']:.3g}, in band {k['band']:.3g}, above band {k['above']:.3g}; "
         f"out-of-band peaks above {KAPPA_MAX:g}: {len(w['peaks'])}")
    if w['peaks']:
        top = sorted(w['peaks'], key=lambda p: -p[1])[:5]
        s += ' (' + ', '.join(f'{f:.3g} Hz: {kv:.3g}' for f, kv in top) + ')'
    s += f"; DC: Z {'finite' if w['Z_dc_exists'] else 'infinite (no DC path)'}"
    return s


def plot_wide(w, path, f_target=None, H_target=None, f_lo=None, f_hi=None, title=''):
    """|diag Z/z0|, |diag Y z0| (one colour per port) and the curvature over the wide grid; target as dots."""
    f = w['f']
    P = w['H']['Z'].shape[1]
    fig, ax = plt.subplots(3, 1, figsize=(10, 10), sharex=True, gridspec_kw={'height_ratios': [2, 2, 1.2]})
    cols = plt.rcParams['axes.prop_cycle'].by_key()['color']
    for r, k in enumerate('ZY'):
        for p in range(P):
            ax[r].loglog(f, np.abs(w['H'][k][:, p]), color=cols[p % len(cols)], lw=1.2, label=f'fit {k}{p + 1}{p + 1}')
            if H_target is not None:
                ax[r].loglog(f_target, np.abs(H_target[k][:, p]), '.', ms=3, color=cols[p % len(cols)])
        ax[r].set_ylabel(f'|{k}n ii|')
    ax[0].legend(ncol=P, fontsize=8, title='lines: fit, dots: target', title_fontsize=8)
    ax[2].loglog(f[1:-1], np.maximum(w['kappa'], 1e-6), 'k', lw=1)
    ax[2].axhline(KAPPA_MAX, color='r', ls='--', lw=1, label=f'curvature {KAPPA_MAX:g} (critically damped pair)')
    ax[2].set_ylabel('curvature |d²ln H/d(ln f)²|')
    ax[2].set_xlabel('frequency (Hz)')
    ax[2].legend(fontsize=8)
    for a in ax:
        if f_lo is not None:
            a.axvspan(f_lo, f_hi, color='0.9', zorder=0)
        a.grid(True, which='both', alpha=0.3)
    ax[0].set_title(title + ' (grey: target band)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
