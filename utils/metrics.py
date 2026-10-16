"""Fit-quality metrics of a fitted S matrix against its target.

For each of S, Zn (Z/z0), Yn (Y*z0), over the unique entries (upper triangle) and
the positive frequencies:
    mae_db      mean |dB(fit) - dB(target)|
    p95_db      95th percentile of the same
    lf_mae_db   mean over the lowest decade of the target (low band)
    hf_mae_db   mean over the highest decade of the target (high band)
    ph_deg      mean |phase error| in degrees (S: weighted by min(|S_target|, 1))
    clog_db     mean 8.686 |ln(fit/target)| (complex log: magnitude AND phase error in one
                dB-equivalent number, 1 rad = 8.686 dB, 10 deg ~ 1.5 dB); the main score
    ripple_db   mean |d2 dB(fit) - d2 dB(target)| along frequency (second difference
                per sample): spurious wiggles/notches the target does not have. Smooth
                offsets do not count, artifacts do.
Frequencies where the target does not exist (NaN, e.g. Z of isolated conductors) are left out.
plus, if the sweep has a DC point and the target exists there: dc_rel, max relative error at DC, and dc_clog_db, mean
8.686 |ln(fit/target)| over the unique entries at DC (not part of the score, which averages positive frequencies).
"""
import numpy as np
import torch

BAND_DECADES = 1.0   # low / high band: the lowest / highest decade of the target's positive frequencies


def _db(x):
    return 20 * np.log10(np.abs(x) + 1e-30)


def _kind_metrics(f, T, P, kind):
    n = T.shape[-1]
    iu = np.triu_indices(n)
    # positive frequencies where the target exists (a Z or Y target can be missing: circuit_model.target_normalized);
    # the low and high bands are the lowest and highest decade of the whole sweep
    pos = (f > 0) & np.isfinite(T).all(axis=(-2, -1))
    t, p = T[pos][:, iu[0], iu[1]], P[pos][:, iu[0], iu[1]]
    fp, f_all = f[pos], f[f > 0]
    lf, hf = fp <= f_all.min() * 10 ** BAND_DECADES, fp >= f_all.max() / 10 ** BAND_DECADES
    e = np.abs(_db(p) - _db(t))
    ph = np.abs(np.angle(p * np.conj(t), deg=True))
    if kind == 'S':
        ph = ph * np.minimum(np.abs(t), 1.0)
    d2 = lambda x: np.diff(x, n=2, axis=0)
    clog = 8.686 * np.abs(np.log((p + 1e-300) / (t + 1e-300)))
    out = {
        'clog_db': float(clog.mean()),
        'lf_clog_db': float(clog[lf].mean()) if (lf).any() else float('nan'),
        'hf_clog_db': float(clog[hf].mean()) if (hf).any() else float('nan'),
        'mae_db': float(e.mean()),
        'p95_db': float(np.percentile(e, 95)),
        'lf_mae_db': float(e[lf].mean()) if (lf).any() else float('nan'),
        'hf_mae_db': float(e[hf].mean()) if (hf).any() else float('nan'),
        'ph_deg': float(ph.mean()),
        'ripple_db': float(np.abs(d2(_db(p)) - d2(_db(t))).mean()),
    }
    if (f == 0).any() and np.isfinite(T[f == 0]).all():
        i0 = int(np.nonzero(f == 0)[0][0])
        out['dc_rel'] = float((np.abs(P[i0] - T[i0]) / (np.abs(T[i0]) + 1e-3 * np.abs(T[i0]).max())).max())
        t0, p0 = T[i0][iu[0], iu[1]], P[i0][iu[0], iu[1]]
        nz = t0 != 0   # an entry exactly 0 in the target has no logarithm
        out['dc_clog_db'] = float((8.686 * np.abs(np.log((p0[nz] + 1e-300) / t0[nz]))).mean()) if nz.any() else 0.0
    return out


def fit_metrics(freqs, S_target, S_pred, extra=None):
    """freqs (N,), S_target/S_pred (N, n, n) tensors or arrays; extra: {'Z': Zn_target,
    'Y': Yn_target} normalized targets (the prediction is derived from S_pred)."""
    import circuit_model
    to_np = lambda x: x.detach().cpu().numpy() if torch.is_tensor(x) else np.asarray(x)
    f = to_np(freqs)
    res = {'S': _kind_metrics(f, to_np(S_target), to_np(S_pred), 'S')}
    for kind, T in (extra or {}).items():
        P = circuit_model.s_to_normalized(torch.as_tensor(S_pred), kind)
        res[kind] = _kind_metrics(f, to_np(T), to_np(P), kind)
    return res


def one_line(m):
    return '  '.join(f"{k}: clog {v['clog_db']:.2f} mae {v['mae_db']:.2f} p95 {v['p95_db']:.2f} lf {v['lf_mae_db']:.2f} "
                     f"hf {v['hf_mae_db']:.2f} rip {v['ripple_db']:.3f}" for k, v in m.items())


def score(m):
    """The study's single number: mean over S, Zn, Yn of clog_db (dB, lower is better)."""
    return float(np.mean([v['clog_db'] for v in m.values()]))
