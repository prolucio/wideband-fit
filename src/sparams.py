"""Network-parameter utilities: numeric setup, S -> Z / Y conversion (Eq. 1) and the target file.

Z and Y are always handled normalised, Z/Z0 and Y*Z0: the losses and errors of the paper are insensitive to this scale.
"""
import os
import warnings

import numpy as np
import torch

DTYPE = torch.float64        # the nodal solve spans many decades of admittance: float64 throughout
CDTYPE = torch.complex128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    # cuSOLVER (not MAGMA) for the batched solve: MAGMA syncs with the host and cannot be captured in a CUDA graph
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.backends.cuda.preferred_linalg_library('cusolver')


def set_deterministic():
    """Deterministic GPU kernels: rerunning the same command on the same hardware and software is bit-identical."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)


def lin_solve(A, B):
    """torch.linalg.solve without the error check (which syncs with the host and blocks CUDA-graph capture)."""
    return torch.linalg.solve_ex(A, B, check_errors=False)[0]


def s_to_z(S):
    """Z / Z0 = (I - S)^-1 (I + S)   (Eq. 1). Batched over leading dimensions."""
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device).expand_as(S)
    return lin_solve(eye - S, eye + S)


def s_to_y(S):
    """Y * Z0 = (I + S)^-1 (I - S)   (Eq. 1)."""
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device).expand_as(S)
    return lin_solve(eye + S, eye - S)


CONVERT = {'S': lambda S: S, 'Z': s_to_z, 'Y': s_to_y}   # the three domains of the fit

# A target Z or Y does not exist at a frequency where its S -> Z / Y conversion is ill-defined (see target_zy)
RCOND_MIN = 1e-12         # relative smallest singular value of I -/+ S below which the conversion is singular
PASSIVE_TOL = 1e-6        # relative negative eigenvalue of Herm(Z), Herm(Y) tolerated: more is rounding noise
G_INSULATION = 1e-9       # S: a conductance to the reference below this is insulation (solver noise), not a path


def target_zy(S, kind, freq, z0=50.0):
    """Z/Z0 ('Z') or Y*Z0 ('Y') of the target S (N_f, P, P), with NaN at the frequencies where it does not exist:
      - I - S (for Z) or I + S (for Y) is singular to working precision (smallest singular value below RCOND_MIN times
        the largest), or
      - the result is not passive (smallest eigenvalue of its Hermitian part below -PASSIVE_TOL times the largest in
        magnitude): it is then the rounding noise of the data around a singular matrix, not a response, or
      - (Z at f = 0 only) some conductor reaches the reference through less than G_INSULATION: a capacitor is open at
        DC, so Z(0) exists only through conduction, and a solver's insulated conductors keep a conductance at its
        rounding level (about 4e-12 S), which would give a finite but meaningless Z(0)."""
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device).expand_as(S)
    sv = torch.linalg.svdvals(eye - S if kind == 'Z' else eye + S)
    H = CONVERT[kind](S)
    ev = torch.linalg.eigvalsh((H + H.conj().transpose(-2, -1)) / 2)
    missing = (sv[..., -1] < RCOND_MIN * sv[..., 0]) | ~torch.isfinite(ev).all(-1) | (
        ev[..., 0] < -PASSIVE_TOL * ev.abs().amax(-1))
    if kind == 'Z':
        dc = torch.as_tensor(freq, device=S.device) == 0
        if dc.any():
            Y = s_to_y(S[dc]) / z0
            g_min = torch.linalg.eigvalsh((Y + Y.conj().transpose(-2, -1)).real / 2)[..., 0]
            missing[dc] |= g_min < G_INSULATION
    return torch.where(missing[..., None, None], torch.full_like(S, float('nan')), H)


class Target:
    """The data to fit: frequencies (Hz, may start with 0), S (N_f, P, P) and the reference impedance z0 (Ohm), plus the
    Z and Y derived from S (NaN where they do not exist). All on the CPU."""

    def __init__(self, freq, S, z0):
        self.freq = np.asarray(freq, dtype=float)
        self.S = torch.as_tensor(S, dtype=CDTYPE)
        self.z0 = float(z0)
        f = torch.as_tensor(self.freq)
        self.Z = target_zy(self.S, 'Z', f, self.z0)
        self.Y = target_zy(self.S, 'Y', f, self.z0)
        self.ports = self.S.shape[-1]
        positive = self.freq[self.freq > 0]
        self.f_min, self.f_max = float(positive.min()), float(positive.max())

    def domains(self):
        """{'S': ..., 'Z': ..., 'Y': ...} target matrices."""
        return {'S': self.S, 'Z': self.Z, 'Y': self.Y}


def load_target(path):
    """Target from the .npz written by parse_touchstone.py (fields freq, params = S, z0)."""
    d = np.load(path)
    return Target(d['freq'], d['params'], d['z0'])


def conductor_groups(target):
    """Groups of ports whose A terminals are galvanically connected (they lie on one conductor), read from the target at
    its lowest frequency (DC if present, where conduction dominates any capacitive coupling): ports p and q are in one
    group when |Y_pq| / sqrt(|Y_pp Y_qq|) > 0.5 (about 1 for the two ends of one trace, ~0 between isolated traces).
    Groups are merged one pair after the other. Returns a list of lists of 0-based ports, or None if Y does not exist at
    the lowest frequency (then every port is its own conductor)."""
    S0 = target.S[int(np.argmin(target.freq))].numpy()
    n = S0.shape[-1]
    try:
        Y0 = np.linalg.solve(np.eye(n) + S0, np.eye(n) - S0)
    except np.linalg.LinAlgError:
        return None
    if not np.isfinite(Y0).all():
        return None
    d = np.sqrt(np.abs(np.diag(Y0)))
    k = np.abs(Y0) / (np.outer(d, d) + 1e-300)
    group = list(range(n))
    for p in range(n):
        for q in range(p + 1, n):
            if k[p, q] > 0.5:
                old, new = group[q], group[p]
                group = [new if g == old else g for g in group]
    return [[p for p in range(n) if group[p] == g] for g in sorted(set(group))]
