"""Training loss (Sec. III-C) and out-of-band penalty (Sec. III-D)."""
import math

import torch

from sparams import CONVERT, DEVICE, DTYPE, target_zy

PHI = {'S': 0.1, 'Z': 1e-3, 'Y': 1e-3}      # floor phi of the relative loss (Table I)


class Loss:
    """Relative loss L_rel (Eq. 8) or logarithmic loss L_ln (Eq. 9) of a netlist's S against the target, summed over the
    domains h in {S, Z, Y}. Per domain and frequency it is the mean over the entries (i, j) of the matrix; the training
    loss is the mean over the frequencies.

    L_rel: |h_hat - h| / (|h| + phi max_ij |h|), the maximum over the entries of the matrix at that frequency.
    L_ln : |ln(h_hat / h)| = sqrt(ln^2 |h_hat / h| + arg^2 (h_hat / h)), over the entries that are not exactly 0 in the
           target (a solver writes exact zeros for the coupling it separates at DC; those entries only enter L_rel).
    A Z or Y target that does not exist at a frequency (sparams.target_zy) is left out of the loss there.
    S_target: (N_f, P, P) on the compute device; freq: (N_f,) its frequencies."""

    def __init__(self, S_target, freq, z0=50.0):
        self.terms = []
        for domain, h in (('S', S_target), ('Z', target_zy(S_target, 'Z', freq, z0)),
                          ('Y', target_zy(S_target, 'Y', freq, z0))):
            exists = torch.isfinite(h).all(dim=(-2, -1))
            h = torch.where(exists[:, None, None], h, torch.ones_like(h))
            mag = h.abs()
            denominator = mag + PHI[domain] * mag.amax(dim=(-2, -1), keepdim=True)
            nonzero = (mag > 0).to(mag.dtype)
            self.terms.append((domain, h, denominator, torch.log(mag + 1e-300), exists.to(mag.dtype), nonzero))

    def __call__(self, S, kind):
        """Loss at every frequency, (N_f,), of the netlist's S; kind 'rel' or 'ln'."""
        h_hat = {'S': S, 'Z': CONVERT['Z'](S), 'Y': CONVERT['Y'](S)}
        out = 0
        for domain, h, denominator, log_mag, exists, nonzero in self.terms:
            if kind == 'rel':
                e = ((h_hat[domain] - h).abs() / denominator).mean(dim=(-2, -1))
            else:
                log_ratio = torch.log(h_hat[domain].abs() + 1e-300) - log_mag
                phase = torch.angle(h_hat[domain] * h.conj())
                e = (torch.sqrt(log_ratio ** 2 + phase ** 2 + 1e-30) * nonzero).sum(dim=(-2, -1)) \
                    / nonzero.sum(dim=(-2, -1)).clamp(min=1)
            out = out + e * exists
        return out


class OutOfBandPenalty:
    """Penalty P on resonant behaviour outside the target band [f_min, f_max], where there is no data (Eqs. 10-11).

    Each step draws K frequencies f, log-uniform over [1 Hz, f_min / MARGIN] and [MARGIN f_max, 10 GHz] (the margin leaves
    room for resonances just outside the band, whose tails reach into it), each with its neighbours f e^-delta and
    f e^delta. The curvature kappa = |d^2 ln h / d(ln f)^2| of the diagonal entries of Z and Y of the netlist is
    estimated by a second difference. A real pole or zero bends ln h by at most 1/2, a critically damped pair by 1, a
    resonance of quality Q by about 4 Q^2. The penalty is the mean of max(kappa - 1, 0).
    The frequencies are redrawn in place by resample() before every step (the CUDA graph reads self.freq) and are shared
    by all seeds trained together. Z and Y only (S is not used)."""

    K = 32               # frequencies per step
    DELTA = 0.01
    KAPPA_MAX = 1.0
    MARGIN = 3.0
    F_RANGE = (1.0, 1e10)    # Hz

    def __init__(self, f_min, f_max):
        d, m = self.DELTA, math.log(self.MARGIN)
        segments = [(math.log(self.F_RANGE[0]) + d, math.log(f_min) - m - d),       # ln f of the centre frequency f
                    (math.log(f_max) + m + d, math.log(self.F_RANGE[1]) - d)]
        self.segments = [(a, b) for a, b in segments if b > a]     # an empty one: the band starts below 3 Hz
        if not self.segments:
            raise ValueError('out-of-band penalty: no frequency range outside the target band')
        self.widths = torch.tensor([b - a for a, b in self.segments], dtype=DTYPE, device=DEVICE)
        self.starts = torch.tensor([a for a, _ in self.segments], dtype=DTYPE, device=DEVICE)
        self.freq = torch.empty(3 * self.K, dtype=DTYPE, device=DEVICE)
        self.resample()

    def resample(self):
        x = torch.rand(self.K, dtype=DTYPE, device=DEVICE) * self.widths.sum()    # position along the segments
        cum = torch.cumsum(self.widths, 0)
        i = torch.searchsorted(cum, x, right=True).clamp(max=len(self.segments) - 1)
        u = self.starts[i] + x - (cum[i] - self.widths[i])
        self.freq.copy_(torch.exp(torch.stack([u - self.DELTA, u, u + self.DELTA], 1)).flatten())

    def __call__(self, S):
        """P for the S (3K, P, P) of the netlist at self.freq."""
        kappa = []
        for domain in 'ZY':
            h = torch.diagonal(CONVERT[domain](S), dim1=-2, dim2=-1).reshape(self.K, 3, -1)
            kappa.append((torch.log(h[:, 2] / h[:, 1]) - torch.log(h[:, 1] / h[:, 0])).abs() / self.DELTA ** 2)
        return torch.relu(torch.cat(kappa, -1) - self.KAPPA_MAX).mean()
