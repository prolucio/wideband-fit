"""Trainable passive RLC network: topology, parameters, element-value constraints, forward model (S matrix), SPICE export.

Node layout: [P1A..PnA | P1B..PnB | resonant-block private nodes | DC-path private nodes | GND].
Port layout (same network either way):
  floating  n ports; port p is the terminal pair (PpA, PpB); P1B is tied to GND, the reference node.
  grounded  2n ports, one per terminal (P1A..PnA, P1B..PnB), each against GND, a separate node (the field solver's
            reference); grounded_to_floating() turns such an S into the floating n-port.

  resonant block  one series R-L-C branch on every pair of nodes among the terminals, its private nodes and GND
  DC path         one series R-L edge on every pair of nodes among the terminals, its own private nodes and GND
  leakage         a fixed conductance from every node to GND (keeps the nodal matrix regular at DC)

Trained parameters (log10 of element values): rlc_logL, rlc_logC, rlc_logR per resonant branch; dc_logR, dc_logL per
DC edge. Every branch is kept inside the window  L <= l_max, C <= c_max, r_min <= R <= r_max, Q = sqrt(L/C) / R <= q_max, and every
DC edge has its corner R / (2 pi L) <= dc_fl_max, by projection after each optimizer step (project_params_); the
forward pass applies the same clamps. Positive R, L, C only: every parameter value is a passive, reciprocal network.

Initial DC path (dc_metal): metal (R low) between two nodes on the same body, dielectric (R high, off but trainable)
between nodes on different bodies. Bodies: the conductors (the A terminals of one conductor) and the ground plate (the B
terminals, GND and any private nodes, the common reference).
With grounded ports, GND is the solver's reference, insulated from every conductor: dielectric to every node.
"""
import math
import os
import warnings

import torch

DTYPE = torch.float64        # the nodal solve spans many decades of admittance: float64 throughout
CDTYPE = torch.complex128
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
if DEVICE == 'cuda':
    # cuSOLVER (not MAGMA) for the batched solve: MAGMA syncs with the host and cannot be captured in a CUDA graph
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        torch.backends.cuda.preferred_linalg_library('cusolver')

Z_REF = 1e2           # Ohm: centre of the initial characteristic-impedance draw
DC_FL_MIN = 1e4      # Hz: lower end of the initial DC-edge corner draw (upper end dc_fl_max)
Q_INIT_MIN, Q_INIT_MAX = 0.05, 30.0   # range of the initial Q draw (capped by q_max)
LOG_L_FLOOR, LOG_C_FLOOR = -16.0, -20.0   # numerical floors of log10 L (H) and log10 C (F)
LOG_R_METAL, LOG_R_DIELECTRIC = -1.0, 13.0   # centres of the initial log10 R (Ohm) draw of the DC-path edges


def set_deterministic():
    """Deterministic GPU kernels: a rerun of the same command on the same hardware/software is bit-identical."""
    os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    torch.use_deterministic_algorithms(True)


class Circuit(torch.nn.Module):
    def __init__(self, ports, f_lo, f_hi, rlc_nodes=16, dc_nodes=0, dc_fl_max=1e6, l_max=1.0, c_max=1e-3,
                 r_min=1e-2, r_max=1e15, q_max=30.0, leak=1e-12, dc_path=True, conductors=None,
                 grounded=False):
        """f_lo, f_hi: positive band of the target (initial resonance draw: f0 log-uniform over it).
        dc_path: False removes the DC path (no private nodes, no edges): every edge is then a series RLC branch.
        conductors: groups of ports (0-based) whose A terminals lie on one conductor (initial DC path: metal between
        them); None: every A terminal on its own conductor.
        grounded: 2 * ports ports, one per terminal against a separate GND node (module docstring); False: floating."""
        super().__init__()
        if not 0 < f_lo < f_hi:
            raise ValueError(f'need 0 < f_lo < f_hi, got {f_lo}, {f_hi}')
        dc_nodes = dc_nodes if dc_path else 0
        self.config = dict(ports=ports, f_lo=float(f_lo), f_hi=float(f_hi), rlc_nodes=rlc_nodes, dc_nodes=dc_nodes,
                           dc_fl_max=float(dc_fl_max), l_max=float(l_max), c_max=float(c_max), r_min=float(r_min),
                           r_max=float(r_max), q_max=float(q_max), leak=float(leak), dc_path=bool(dc_path),
                           grounded=bool(grounded))
        self.ports, self.f_lo, self.f_hi = ports, float(f_lo), float(f_hi)
        self.rlc_nodes, self.dc_nodes, self.dc_fl_max = rlc_nodes, dc_nodes, float(dc_fl_max)
        self.log_lmax, self.log_cmax = math.log10(l_max), math.log10(c_max)
        self.log_rmin, self.log_rmax = math.log10(r_min), math.log10(r_max)
        self.q_max = float(q_max)
        self.grounded = bool(grounded)
        self.n_out = 2 * ports if grounded else ports   # ports of the S matrix

        self.n_terminals = 2 * ports
        self.rlc_start = self.n_terminals
        self.dc_start = self.rlc_start + rlc_nodes
        self.gnd = self.dc_start + dc_nodes
        self.n_active = self.gnd   # the GND row/column is dropped from the solve

        terminals = list(range(self.n_terminals))
        rlc_pairs = complete_graph(terminals + list(range(self.rlc_start, self.dc_start)) + [self.gnd])
        dc_pairs = complete_graph(terminals + list(range(self.dc_start, self.gnd)) + [self.gnd]) if dc_path else []
        self.n_rlc, self.n_dc = len(rlc_pairs), len(dc_pairs)
        pairs = torch.tensor(rlc_pairs + dc_pairs, dtype=torch.long, device=DEVICE)

        buf = lambda name, t: self.register_buffer(name, t, persistent=False)
        lng = lambda x: torch.tensor(x, dtype=torch.long, device=DEVICE)
        buf('pair_a', pairs[:, 0])
        buf('pair_b', pairs[:, 1])
        # floating: P1B is the reference, its stamps land on GND (node index map, GND = last); grounded: GND is its
        # own node and every terminal is a port against it
        node_map = torch.arange(self.gnd + 1, dtype=torch.long, device=DEVICE)
        if not grounded:
            node_map[ports] = self.gnd
        buf('node_map', node_map)
        n = self.gnd + 1
        a, b = node_map[self.pair_a], node_map[self.pair_b]
        if grounded:
            pa, pb = node_map[:self.n_terminals], torch.full((self.n_terminals,), self.gnd, dtype=torch.long,
                                                              device=DEVICE)
        else:
            pa, pb = node_map[:ports], node_map[ports:self.n_terminals]
        buf('port_a', pa)
        buf('port_b', pb)
        # flat indices into the (n x n) nodal matrix: +Y at (a,a), (b,b), -Y at (a,b), (b,a) per edge, then the
        # port terminations z0 between PpA and PpB
        buf('stamp_idx', torch.cat([a * n + a, b * n + b, a * n + b, b * n + a,
                                    pa * n + pa, pb * n + pb, pa * n + pb, pb * n + pa]))
        ones = torch.ones(len(pairs), dtype=DTYPE, device=DEVICE)
        buf('stamp_sign', torch.cat([ones, ones, -ones, -ones]))
        # nodes kept when the resonant block is off (DC-path pretrain): everything but its private nodes
        buf('keep_no_rlc', lng([i for i in range(self.gnd) if not self.rlc_start <= i < self.dc_start]))
        buf('leak', torch.eye(self.n_active, dtype=CDTYPE, device=DEVICE) * leak)
        # element masks (False = removed by slim.py), saved in checkpoints
        self.register_buffer('branch_alive', torch.ones(self.n_rlc, dtype=torch.bool, device=DEVICE))
        self.register_buffer('dc_alive', torch.ones(self.n_dc, dtype=torch.bool, device=DEVICE))
        self._init_parameters(conductors)

    # ------------------------------------------------------------------------------------------- initialisation

    def dc_metal(self, conductors=None):
        """(n_dc,) bool: DC edges that start as metal. Dielectric: an A terminal to a node that is not an A terminal
        (trace-to-reference insulation), and between A terminals of different conductors. Metal: everything else
        (B terminals, private nodes and GND: the common reference; A terminals of one conductor). Grounded ports: GND
        is the solver's reference, insulated from every conductor, so every edge to GND is dielectric too."""
        group = list(range(self.ports))   # conductor index of each port's A terminal
        for g in conductors or []:
            for p in g:
                group[p] = min(g)
        a, b = self.pair_a[self.n_rlc:].tolist(), self.pair_b[self.n_rlc:].tolist()
        is_a = lambda i: i < self.ports
        metal = [((is_a(x) and is_a(y) and group[x] == group[y]) or not (is_a(x) or is_a(y)))
                 and not (self.grounded and self.gnd in (x, y)) for x, y in zip(a, b)]
        return torch.tensor(metal, dtype=torch.bool, device=DEVICE)

    def _init_parameters(self, conductors):
        """Resonant branches drawn as (f0, z = sqrt(L/C), Q) and converted to (L, C, R); DC edges as (R, corner fL),
        metal or dielectric (dc_metal)."""
        n = self.n_rlc
        lo, hi = math.log10(self.f_lo), math.log10(self.f_hi)
        log_f0 = lo + (hi - lo) * torch.rand(n, dtype=DTYPE, device=DEVICE)   # log-uniform over the band
        log_z = torch.randn(n, dtype=DTYPE, device=DEVICE) + math.log10(Z_REF)
        q_hi = min(self.q_max, Q_INIT_MAX)   # log-uniform Q
        q = 10 ** (math.log10(Q_INIT_MIN) + (math.log10(q_hi) - math.log10(Q_INIT_MIN))
                   * torch.rand(n, dtype=DTYPE, device=DEVICE))
        log_w0 = log_f0 + math.log10(2 * math.pi)
        self.rlc_logL = torch.nn.Parameter(log_z - log_w0)
        self.rlc_logC = torch.nn.Parameter(-log_z - log_w0)
        self.rlc_logR = torch.nn.Parameter(log_z - torch.log10(q))
        # DC path: log10 R ~ N(-1, 1) for metal, N(13, 1) for dielectric (far above the leakage: off, but trainable);
        # corner fL log-uniform over [DC_FL_MIN, dc_fl_max], L = R / (2 pi fL)
        log_r = torch.randn(self.n_dc, dtype=DTYPE, device=DEVICE) + torch.where(
            self.dc_metal(conductors), LOG_R_METAL, LOG_R_DIELECTRIC).to(DTYPE)
        lo_fl = math.log10(min(DC_FL_MIN, self.dc_fl_max))
        f_l = 10 ** (lo_fl + (math.log10(self.dc_fl_max) - lo_fl) * torch.rand(self.n_dc, dtype=DTYPE, device=DEVICE))
        self.dc_logR = torch.nn.Parameter(log_r)
        self.dc_logL = torch.nn.Parameter(log_r - torch.log10(2 * torch.pi * f_l))
        with torch.no_grad():
            self.project_params_(dict(self.named_parameters()))

    # ------------------------------------------------------------------------------------ constraints (window)

    def project_params_(self, P):
        """In place: project the parameters in P (name -> tensor, optionally with leading model dims) into the window."""
        lL, lC, lR = P['rlc_logL'], P['rlc_logC'], P['rlc_logR']
        lL.copy_(lL.clamp(min=LOG_L_FLOOR, max=self.log_lmax))
        lC.copy_(lC.clamp(min=LOG_C_FLOOR, max=self.log_cmax))
        r_lo = torch.clamp((lL - lC) / 2 - math.log10(self.q_max), min=self.log_rmin)   # R >= max(r_min, z / q_max)
        lR.copy_(torch.maximum(lR.clamp(max=self.log_rmax), r_lo))
        dR, dL = P['dc_logR'], P['dc_logL']
        dR.copy_(dR.clamp(min=self.log_rmin, max=self.log_rmax))
        l_lo = dR - math.log10(2 * math.pi * self.dc_fl_max)                               # corner fL <= dc_fl_max
        dL.copy_(torch.minimum(torch.maximum(dL, l_lo), torch.full_like(dL, self.log_lmax)).clamp(min=LOG_L_FLOOR))

    # ---------------------------------------------------------------------------------------- element values

    def rlc_values(self):
        """(w0, z = sqrt(L/C), Q) of every resonant branch, clamped to the window."""
        lL = torch.clamp(self.rlc_logL, max=self.log_lmax)
        lC = torch.clamp(self.rlc_logC, max=self.log_cmax)
        log_z = (lL - lC) / 2
        lR = torch.clamp(self.rlc_logR, max=self.log_rmax)
        lR = torch.clamp(lR, min=torch.clamp(log_z - math.log10(self.q_max), min=self.log_rmin).detach())
        return 10 ** (-(lL + lC) / 2), 10 ** log_z, 10 ** (log_z - lR)

    def dc_values(self):
        """(R, L) of every DC-path edge, clamped to the window."""
        log_r = torch.clamp(self.dc_logR, min=self.log_rmin, max=self.log_rmax)
        log_l = torch.clamp(self.dc_logL, min=(log_r - math.log10(2 * math.pi * self.dc_fl_max)).detach())
        log_l = torch.clamp(log_l, max=self.log_lmax)
        return 10 ** log_r, 10 ** log_l

    def rlc_admittances(self, omega):
        """(N_freq, N_branch) series-RLC admittances, written as jw / (z w0 [(1 - (w/w0)^2) + j (w/w0) / Q])
        (= 1 / (R + jwL + 1/(jwC)), finite at w = 0). omega: (N_freq, 1)."""
        w0, z, Q = self.rlc_values()
        ratio = omega / w0
        Y = cplx(torch.zeros_like(omega), omega) / cplx(z * w0 * (1 - ratio ** 2), z * w0 * ratio / Q)
        return Y * self.branch_alive.to(DTYPE)

    def dc_admittances(self, omega):
        R, L = self.dc_values()
        return 1 / cplx(R.expand_as(omega * R), omega * L) * self.dc_alive.to(DTYPE)

    # ------------------------------------------------------------------------------------------- forward model

    def forward(self, freq, z0=50.0, rlc_off=False):
        """S matrix (N_freq, n_out, n_out) at the frequencies freq (Hz). rlc_off: resonant block removed."""
        n_f = freq.shape[0]
        omega = 2 * torch.pi * freq.to(DTYPE).view(-1, 1)
        if rlc_off:
            Y_rlc = torch.zeros(n_f, self.n_rlc, dtype=CDTYPE, device=DEVICE)
        else:
            Y_rlc = self.rlc_admittances(omega)
        Y_edge = torch.cat([Y_rlc, self.dc_admittances(omega)], dim=1)

        # nodal admittance matrix including GND (row/column dropped afterwards), plus the leakage to GND
        n = self.gnd + 1
        y0 = torch.full((n_f, self.n_out), 1.0 / z0, dtype=CDTYPE, device=DEVICE)
        vals = torch.cat([Y_edge.repeat(1, 4) * self.stamp_sign, y0, y0, -y0, -y0], dim=1)
        G = torch.zeros(n_f, n * n, dtype=CDTYPE, device=DEVICE)
        G = G.index_add(1, self.stamp_idx, vals).view(n_f, n, n)[:, :-1, :-1] + self.leak

        # all ports driven at once: current 2/sqrt(z0) into PpA, out of PpB (one multi-RHS solve)
        sqrt_z0 = z0 ** 0.5
        pa, pb = self.port_a, self.port_b
        inc = (torch.nn.functional.one_hot(pa, n) - torch.nn.functional.one_hot(pb, n)).T.to(CDTYPE)
        currents = (inc * (2 / sqrt_z0)).expand(n_f, n, self.n_out)[:, :-1]
        if rlc_off:   # the resonant block's private nodes are then isolated: solve the remaining nodes only
            keep = self.keep_no_rlc
            v_k = lin_solve(G[:, keep][:, :, keep], currents[:, keep])
            v = torch.zeros(n_f, n - 1, self.n_out, dtype=CDTYPE, device=DEVICE).index_copy(1, keep, v_k)
        else:
            v = lin_solve(G, currents)
        v = torch.cat([v, torch.zeros_like(v[:, :1])], dim=1)   # GND at 0 V

        v_ports = v[:, pa, :] - v[:, pb, :]                     # (N_freq, port m, driven port k)
        i_ports = torch.eye(self.n_out, dtype=CDTYPE, device=DEVICE) * (2 / sqrt_z0) - v_ports / z0
        a_waves = (v_ports + z0 * i_ports) / (2 * sqrt_z0)
        b_waves = (v_ports - z0 * i_ports) / (2 * sqrt_z0)
        return b_waves / torch.diagonal(a_waves, dim1=-2, dim2=-1)[:, None, :]

    # --------------------------------------------------------------------------------------------- reporting

    def node_name(self, i):
        if i < self.ports:
            return f'P{i + 1}A'
        if i < self.n_terminals:
            return f'P{i - self.ports + 1}B'
        if i < self.dc_start:
            return f'rlc{i - self.rlc_start}'
        if i < self.gnd:
            return f'dc{i - self.dc_start}'
        return 'GND'

    def summary(self):
        dc = (f'  DC path: {self.dc_nodes} private nodes, {int(self.dc_alive.sum())}/{self.n_dc} series-RL edges, '
              f'corner fL <= {self.dc_fl_max:g} Hz\n' if self.n_dc else '  DC path: none\n')
        ports = (f'{self.n_out} grounded ports ({self.n_terminals} terminals, each against GND)' if self.grounded
                 else f'{self.ports} floating ports ({self.n_terminals} terminals, P1B = reference)')
        return (f'{ports}\n'
                f'  resonant block: {self.rlc_nodes} private nodes, {int(self.branch_alive.sum())}/{self.n_rlc} '
                f'series-RLC branches\n{dc}'
                f'  window: L <= {10 ** self.log_lmax:g} H, C <= {10 ** self.log_cmax:g} F, '
                f'{10 ** self.log_rmin:g} <= R <= {10 ** self.log_rmax:g} Ohm, Q <= {self.q_max:g}; '
                f'leak {float(self.leak[0, 0].real):g} S\n'
                f'  trainable parameters: {sum(p.numel() for p in self.parameters())}')

    @torch.no_grad()
    def write_spice(self, path, name='fitted', z0=50.0):
        """SPICE subcircuit with pins P1A P1B ... PnA PnB (port p: PpA + to PpB -; grounded: plus GND, every pin a port
        against it); alive elements only, plus the leak resistors of the solver (they also give every node a DC path
        to the reference)."""
        gnd_name = self.node_name(self.gnd if self.grounded else self.ports)   # floating: P1B is the reference node
        node = lambda i: self.node_name(i) if self.grounded or i not in (self.gnd, self.ports) else gnd_name
        fmt = lambda v: f'{float(v):.12g}'
        lines, used = [], set()
        w0, z, Q = (t.cpu() for t in self.rlc_values())
        alive = self.branch_alive.cpu().tolist()
        for e in range(self.n_rlc):
            a, b = node(int(self.pair_a[e])), node(int(self.pair_b[e]))
            if not alive[e] or a == b:
                continue
            used.update((a, b))
            lines += [f'* RLC[{e}] {a}-{b}', f'Rrlc{e} {a} nrlc{e}_1 {fmt(z[e] / Q[e])}',
                      f'Lrlc{e} nrlc{e}_1 nrlc{e}_2 {fmt(z[e] / w0[e])}', f'Crlc{e} nrlc{e}_2 {b} {fmt(1 / (z[e] * w0[e]))}']
        R, L = (t.cpu() for t in self.dc_values())
        dc_alive = self.dc_alive.cpu().tolist()
        for e in range(self.n_dc):
            a, b = node(int(self.pair_a[self.n_rlc + e])), node(int(self.pair_b[self.n_rlc + e]))
            if not dc_alive[e] or a == b:
                continue
            used.update((a, b))
            lines += [f'* DC[{e}] {a}-{b}', f'Rdc{e} {a} ndc{e}_1 {fmt(R[e])}', f'Ldc{e} ndc{e}_1 {b} {fmt(L[e])}']
        pins = [self.node_name(x) for p in range(self.ports) for x in (p, p + self.ports)] + ['GND'] * self.grounded
        leak = float(self.leak[0, 0].real)
        leaks = [f'Rleak_{m} {m} {gnd_name} {fmt(1 / leak)}'
                 for m in sorted(used | {node(t) for t in range(self.n_terminals)}) if m != gnd_name]
        header = [f'* {name}: {self.n_out}-port fitted model, reference z0 = {z0:g} Ohm',
                  '* every pin is a port against GND' if self.grounded else
                  f'* port p is PpA (+) to PpB (-); the reference node is {gnd_name}',
                  f'.subckt {name} ' + ' '.join(pins)]
        with open(path, 'w') as fh:
            fh.write('\n'.join(header + ['* ---- branches'] + lines + ['* ---- leaks'] + leaks + [f'.ends {name}', '']))


# ------------------------------------------------------------------------------------------------- helpers

def complete_graph(nodes):
    return [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]


def cplx(re, im):
    """torch.complex(re, im); under torch.func.vmap (no batching rule for torch.complex) built arithmetically."""
    if torch._C._functorch.is_batchedtensor(re) or torch._C._functorch.is_batchedtensor(im):
        return re + 1j * im
    return torch.complex(re, im)


def lin_solve(A, B):
    """torch.linalg.solve without the error check (which syncs with the host and blocks CUDA-graph capture)."""
    return torch.linalg.solve_ex(A, B, check_errors=False)[0]


def s_to_normalized(S, kind):
    """S -> Z/z0 ('Z') = (I - S)^-1 (I + S), or Y z0 ('Y') = (I + S)^-1 (I - S). Batched over leading dims."""
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device).expand_as(S)
    if kind == 'Z':
        return lin_solve(eye - S, eye + S)
    if kind == 'Y':
        return lin_solve(eye + S, eye - S)
    raise ValueError(f"kind must be 'Z' or 'Y', got {kind!r}")


TARGET_RCOND = 1e-12
TARGET_PASSIVE_TOL = 1e-6
G_INSULATION = 1e-9   # S: below this, a conductance to the reference at DC is insulation (solver noise), not a path


def target_normalized(S, kind, freq=None, z0=50.0):
    """Z/z0 ('Z') or Y z0 ('Y') of a target S, NaN at the frequencies where it does not exist: I - S (for Z) or I + S
    (for Y) singular to working precision (smallest / largest singular value below TARGET_RCOND), or the result not
    passive (smallest eigenvalue of its Hermitian part below -TARGET_PASSIVE_TOL times the largest in magnitude): then
    it is the rounding noise of the data around a singular matrix, not a response. Example: Z at DC of a structure
    whose conductors are isolated from each other (infinite open-circuit impedance; the Q3D 4-port gives -3.2e9 z0).
    freq (Hz, optional): Z at DC also does not exist when some conductor reaches the reference only through less than
    G_INSULATION (the smallest eigenvalue of the conductance matrix Y(0) below it): a capacitor is open at DC, so Z(0)
    exists only through conduction, and a solver's isolated conductors leave a conductance at its rounding level
    (Q3D: about 4e-12 S, which gives a finite but meaningless Z(0) with Z13 exactly 0)."""
    eye = torch.eye(S.shape[-1], dtype=S.dtype, device=S.device).expand_as(S)
    sv = torch.linalg.svdvals(eye - S if kind == 'Z' else eye + S)
    H = s_to_normalized(S, kind)
    ev = torch.linalg.eigvalsh((H + H.conj().transpose(-2, -1)) / 2)
    missing = (sv[..., -1] < TARGET_RCOND * sv[..., 0]) | ~torch.isfinite(ev).all(-1) | (
        ev[..., 0] < -TARGET_PASSIVE_TOL * ev.abs().amax(-1))
    if kind == 'Z' and freq is not None:
        dc = torch.as_tensor(freq, device=S.device) == 0
        if dc.any():
            Yd = s_to_normalized(S[dc], 'Y') / z0
            g_min = torch.linalg.eigvalsh((Yd + Yd.conj().transpose(-2, -1)).real / 2)[..., 0]
            missing[dc] |= g_min < G_INSULATION
    return torch.where(missing[..., None, None], torch.full_like(S, float('nan')), H)


def grounded_to_floating(S, z0=50.0, leak=1e-12):
    """Floating n-port S (..., n, n) of a grounded 2n-port S (..., 2n, 2n) (terminals ordered A1..An, B1..Bn, each a
    port against GND): port p becomes the pair (Ap, Bp) and GND is left floating (it keeps its admittances to the
    terminals). The terminal network is the indefinite admittance matrix of Y with GND added; a conductance leak
    between GND and B1 keeps it regular at DC (as the Circuit leakage), and B1 is the reference node. Batched."""
    n2 = S.shape[-1]
    n = n2 // 2
    eye2 = torch.eye(n2, dtype=S.dtype, device=S.device)
    Y = lin_solve(eye2 + S, eye2 - S)                                   # Y z0
    Yi = torch.zeros(S.shape[:-2] + (n2 + 1, n2 + 1), dtype=S.dtype, device=S.device)
    Yi[..., :n2, :n2] = Y
    Yi[..., n2, :n2] = -Y.sum(-2)
    Yi[..., :n2, n2] = -Y.sum(-1)
    Yi[..., n2, n2] = Y.sum((-2, -1)) + leak * z0
    Yi[..., n, n] += leak * z0
    Yi[..., n, n2] -= leak * z0
    Yi[..., n2, n] -= leak * z0
    K = torch.zeros(n2 + 1, n, dtype=S.dtype, device=S.device)         # port incidence: +1 at Ap, -1 at Bp
    K[torch.arange(n), torch.arange(n)] = 1
    K[torch.arange(n) + n, torch.arange(n)] = -1
    keep = [i for i in range(n2 + 1) if i != n]                         # B1 is the reference
    G = (Yi + K @ K.T)[..., keep, :][..., keep]                         # plus the port terminations (z0)
    V = lin_solve(G, (2 * K[keep]).expand(S.shape[:-2] + (n2, n)))
    return K[keep].T @ V - torch.eye(n, dtype=S.dtype, device=S.device)


def get_db(H, eps=1e-12):
    return 20 * torch.log10(torch.abs(H) + eps)


def get_phase(H):
    return torch.atan2(torch.imag(H), torch.real(H))


@torch.no_grad()
def passivity_sweep(circuit, z0=50.0, f_lo=1e-3, f_hi=1e13, per_decade=200, extra_freqs=None, chunk=1000):
    """Passivity margins on a dense grid (DC, then f_lo..f_hi log-spaced, plus extra_freqs): 1 - sigma_max(S)^2 and
    the smallest eigenvalue of the Hermitian part of Y z0. Both >= 0 for a passive network. Returns (f, m_S, m_Y)."""
    n_dec = math.log10(f_hi / f_lo)
    grid = torch.cat([torch.zeros(1, dtype=torch.float64),
                      torch.logspace(math.log10(f_lo), math.log10(f_hi), int(per_decade * n_dec) + 1,
                                     dtype=torch.float64)])
    if extra_freqs is not None:
        grid = torch.unique(torch.cat([grid, torch.as_tensor(extra_freqs, dtype=torch.float64).cpu()]))
    m_s, m_y = [], []
    for part in grid.split(chunk):
        S = circuit(part.to(DEVICE), z0=z0).cpu()
        m_s.append(1 - torch.linalg.svdvals(S)[..., 0] ** 2)
        Y = s_to_normalized(S, 'Y')
        m_y.append(torch.linalg.eigvalsh((Y + Y.conj().transpose(-2, -1)) / 2)[..., 0])
    return grid, torch.cat(m_s), torch.cat(m_y)


def check_passivity(circuit, z0=50.0, tol=1e-9, **kw):
    """Print the passivity margins on the dense grid; True if both stay >= -tol."""
    f, m_s, m_y = passivity_sweep(circuit, z0=z0, **kw)
    ok = bool((m_s.min() >= -tol) and (m_y.min() >= -tol))
    print(f'passivity on {len(f)} frequencies (DC, {f[1]:.3g}-{f[-1]:.3g} Hz): min 1 - sigma_max(S)^2 = '
          f'{m_s.min().item():.3e}, min eig Herm(Y z0) = {m_y.min().item():.3e} -> {"PASS" if ok else "FAIL"}')
    return ok


def load_checkpoint(path):
    """(Circuit, checkpoint dict) from a file written by train.py or slim.py."""
    ck = torch.load(path, map_location=DEVICE)
    c = Circuit(**ck['config'])
    c.load_state_dict(ck['state_dict'])
    return c, ck
