"""Errors of a fitted S matrix against the target, and the passivity check.

For each domain h in {S, Z, Y}, over all frequencies (DC included) and the unique entries (i <= j: the matrices are
symmetric), skipping the entries that are missing (NaN: Z and Y can be missing, see sparams.target_zy) or exactly 0 in the
target (a solver writes exact zeros for the coupling it separates at DC; their dB and phase do not exist):
    magnitude   mean | 20 log10 |h_hat| - 20 log10 |h| |, in dB
    phase       mean | arg(h_hat / h) |, in degrees
The reported errors are the means of the three domains. Ranking is by magnitude error only.
For the pruning test (prune.py) the logarithmic error |ln(h_hat / h)| (magnitude and phase in one number, in neper; 1 neper = 8.686 dB)
is also computed over the positive frequencies: the whole band, its lowest decade, its highest decade, and, apart, at DC.
"""
import math

import numpy as np
import torch

from sparams import CONVERT, DEVICE


def _db(x):
    return 20 * np.log10(np.abs(x) + 1e-30)


def domain_errors(freq, h, h_hat, domain):
    """Errors of one domain; freq (N_f,) in Hz, h and h_hat (N_f, P, P) target and fit (numpy)."""
    iu = np.triu_indices(h.shape[-1])
    t_all, p_all = h[:, iu[0], iu[1]], h_hat[:, iu[0], iu[1]]
    with np.errstate(invalid='ignore'):
        valid = np.isfinite(t_all) & (np.abs(t_all) > 0)
        magnitude = np.abs(_db(p_all) - _db(t_all))[valid]
        phase = np.abs(np.angle(p_all * np.conj(t_all), deg=True))[valid]
    out = {'magnitude': float(magnitude.mean()), 'phase': float(phase.mean())}

    # logarithmic error of the pruning test: positive frequencies where the target exists, and the DC point apart
    exists = (freq > 0) & np.isfinite(h).all(axis=(-2, -1))
    t, p = t_all[exists], p_all[exists]
    f_pos = freq[freq > 0]
    low = freq[exists] <= f_pos.min() * 10
    high = freq[exists] >= f_pos.max() / 10
    log_error = np.abs(np.log((p + 1e-300) / (t + 1e-300)))
    out.update(ln=float(log_error.mean()),
               ln_low=float(log_error[low].mean()) if low.any() else float('nan'),
               ln_high=float(log_error[high].mean()) if high.any() else float('nan'))
    dc = np.nonzero(freq == 0)[0]
    if len(dc) and np.isfinite(h[dc[0]]).all():
        t0, p0 = t_all[dc[0]], p_all[dc[0]]
        nonzero = t0 != 0       # an entry exactly 0 in the target has no logarithm
        out['ln_dc'] = float(np.abs(np.log((p0[nonzero] + 1e-300) / t0[nonzero])).mean()) if nonzero.any() else 0.0
    return out


def fit_errors(target, S_fit):
    """{'S': ..., 'Z': ..., 'Y': ...} -> errors (domain_errors) of the fitted S (N_f, P, P, CPU tensor) against the Target."""
    out = {}
    for domain, h in target.domains().items():
        h_hat = CONVERT[domain](S_fit)
        out[domain] = domain_errors(target.freq, h.numpy(), h_hat.numpy(), domain)
    return out


def magnitude_error(errors):
    """Mean over S, Z, Y of the magnitude error (dB); the number used to rank the fits."""
    return float(np.mean([e['magnitude'] for e in errors.values()]))


def phase_error(errors):
    """Mean over S, Z, Y of the phase error (degrees)."""
    return float(np.mean([e['phase'] for e in errors.values()]))


@torch.no_grad()
def passivity_margins(net, z0=50.0, extra_freqs=None, f_lo=1e-3, f_hi=1e13, per_decade=200, chunk=1000):
    """Smallest values over a dense frequency grid (DC, f_lo..f_hi log-spaced, plus extra_freqs) of 1 - sigma_max(S)^2 and
    of the smallest eigenvalue of the Hermitian part of Y*Z0. Both are >= 0 for a passive network."""
    grid = torch.cat([torch.zeros(1, dtype=torch.float64),
                      torch.logspace(math.log10(f_lo), math.log10(f_hi), int(per_decade * math.log10(f_hi / f_lo)) + 1,
                                     dtype=torch.float64)])
    if extra_freqs is not None:
        grid = torch.unique(torch.cat([grid, torch.as_tensor(extra_freqs, dtype=torch.float64).cpu()]))
    margin_s, margin_y = [], []
    for part in grid.split(chunk):
        S = net(part.to(DEVICE), z0=z0).cpu()
        margin_s.append(1 - torch.linalg.svdvals(S)[..., 0] ** 2)
        Y = CONVERT['Y'](S)
        margin_y.append(torch.linalg.eigvalsh((Y + Y.conj().transpose(-2, -1)) / 2)[..., 0])
    return float(torch.cat(margin_s).min()), float(torch.cat(margin_y).min())


def is_passive(net, z0=50.0, extra_freqs=None, tol=1e-9):
    """True if both passivity margins stay >= -tol (guaranteed by construction; this detects numerical problems)."""
    return min(passivity_margins(net, z0, extra_freqs)) >= -tol
