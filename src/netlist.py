"""Trainable passive netlist (paper, Sec. III-A): topology, element values, S-parameters of the netlist, SPICE export.

Topology (P ports, N internal nodes; Fig. 2 of the paper)
    nodes   0..P-1 the terminals P1A..PPA, P..2P-1 the terminals P1B..PPB, then the N internal nodes. Port p is the
            terminal pair (PpA, PpB). P1B is the 0 V reference node of the netlist.
    RLC     one series R-L-C edge between every pair of the 2P + N nodes: (2P + N)(2P + N - 1) / 2 edges.
    RL      one series R-L edge between every pair of the 2P terminals: P(2P - 1) edges.
    R_eps   a fixed resistor R_EPS from every node to 0 V (keeps the nodal matrix regular at DC).
  With P = 4, N = 16 that is 276 RLC and 28 RL edges, 24 nodes and 23 resistors R_eps.

Trained parameters, per edge n (Eq. 3): r_n, l_n (and c_n for RLC edges), the base-10 logarithms of R (Ohm), L (H), C (F).
They are kept inside the allowed range by clamping in the forward pass (as in Eq. 3) and by projecting them after every
optimiser step (project_), so that no parameter sits where the clamp has zero gradient. Positive R, L, C only: every
parameter value is a passive and reciprocal netlist.
"""
import math

import torch

from sparams import CDTYPE, DEVICE, DTYPE, lin_solve

# Fixed parameters (Table I of the paper)
N_INTERNAL = 16                    # N
R_MIN, R_MAX = 1e-2, 1e15          # Ohm
L_MIN, L_MAX = 1e-16, 1.0          # H
C_MIN, C_MAX = 1e-20, 1e-3         # F
F_DC_CAP = 1e6                     # Hz: cutoff R / (2 pi L) of the RL edges stays below this
Q_CAP = 30.0                       # quality factor sqrt(L / C) / R of the RLC edges stays below this
R_EPS = 1e12                       # Ohm

# Initial draws (Eqs. 6 and 7)
LOG_Z_MEAN, LOG_Z_STD = 2.0, 1.0           # log10 z ~ N(2, 1), z = sqrt(L / C) in Ohm
Q_INIT_RANGE = (0.05, 30.0)                # Q log-uniform
LOG_R_RL_CONNECTED, LOG_R_RL_ISOLATED = -1.0, 13.0   # mean of log10 R of RL edges between connected / isolated nodes
LOG_FL_RANGE = (4.0, 6.0)                  # log10 of the corner frequency f_L = R / (2 pi L) of RL edges, uniform

LOG_R_MIN, LOG_R_MAX = math.log10(R_MIN), math.log10(R_MAX)
LOG_L_MIN, LOG_L_MAX = math.log10(L_MIN), math.log10(L_MAX)
LOG_C_MIN, LOG_C_MAX = math.log10(C_MIN), math.log10(C_MAX)
LOG_2PI_FDC = math.log10(2 * math.pi * F_DC_CAP)    # log10 R - log10 L >= this for RL edges (Eq. 4)
LOG_Q_CAP = math.log10(Q_CAP)


def complete_graph(nodes):
    return [(nodes[i], nodes[j]) for i in range(len(nodes)) for j in range(i + 1, len(nodes))]


def cplx(re, im):
    """torch.complex(re, im); under torch.func.vmap (no batching rule for torch.complex) built arithmetically."""
    if torch._C._functorch.is_batchedtensor(re) or torch._C._functorch.is_batchedtensor(im):
        return re + 1j * im
    return torch.complex(re, im)


class Netlist(torch.nn.Module):
    def __init__(self, ports, f_min, f_max, internal_nodes=N_INTERNAL, groups=None):
        """[f_min, f_max]: positive band of the target (the resonance frequencies f_0 are drawn log-uniformly over it).
        groups: lists of ports (0-based) whose A terminals are galvanically connected (conductor_groups); the RL edges
        between them start as good conductors. None: no two ports are connected."""
        super().__init__()
        if not 0 < f_min < f_max:
            raise ValueError(f'need 0 < f_min < f_max, got {f_min}, {f_max}')
        self.config = dict(ports=ports, f_min=float(f_min), f_max=float(f_max), internal_nodes=internal_nodes)
        self.ports, self.f_min, self.f_max = ports, float(f_min), float(f_max)
        self.n_terminals = 2 * ports
        self.n_nodes = self.n_terminals + internal_nodes
        self.reference = ports                      # P1B

        rlc_pairs = complete_graph(list(range(self.n_nodes)))
        rl_pairs = complete_graph(list(range(self.n_terminals)))
        self.n_rlc, self.n_rl = len(rlc_pairs), len(rl_pairs)
        pairs = torch.tensor(rlc_pairs + rl_pairs, dtype=torch.long, device=DEVICE)

        buf = lambda name, t: self.register_buffer(name, t, persistent=False)
        buf('pair_a', pairs[:, 0])
        buf('pair_b', pairs[:, 1])
        n = self.n_nodes
        a, b = self.pair_a, self.pair_b
        pa = torch.arange(ports, device=DEVICE)
        pb = torch.arange(ports, self.n_terminals, device=DEVICE)
        buf('port_a', pa)
        buf('port_b', pb)
        # flat indices into the (n x n) nodal matrix: +Y at (a,a), (b,b), -Y at (a,b), (b,a) for every edge, then the
        # same for the port terminations Z0 between PpA and PpB
        buf('stamp_idx', torch.cat([a * n + a, b * n + b, a * n + b, b * n + a,
                                    pa * n + pa, pb * n + pb, pa * n + pb, pb * n + pa]))
        ones = torch.ones(len(pairs), dtype=DTYPE, device=DEVICE)
        buf('stamp_sign', torch.cat([ones, ones, -ones, -ones]))
        # nodes whose voltage is solved for: all but the reference; and, with the RLC edges off, the terminals only
        buf('solved_nodes', torch.tensor([i for i in range(n) if i != self.reference], device=DEVICE))
        buf('solved_terminals', torch.tensor([i for i in range(self.n_terminals) if i != self.reference], device=DEVICE))
        # masks of the edges still present (False = removed by pruning); saved in checkpoints
        self.register_buffer('rlc_alive', torch.ones(self.n_rlc, dtype=torch.bool, device=DEVICE))
        self.register_buffer('rl_alive', torch.ones(self.n_rl, dtype=torch.bool, device=DEVICE))
        self._init_parameters(groups)

    # ------------------------------------------------------------------------------------------- initialisation

    def rl_starts_as_metal(self, groups=None):
        """(n_rl,) bool: RL edges whose terminals are galvanically connected (they start as good conductors, Eq. 7).
        Nodes belong to one of two kinds of body: the conductors (the A terminals; the A terminals of the ports in one
        group are one conductor) and the common reference (all B terminals). An edge is metal if it joins two
        A terminals of the same conductor, or two B terminals; it is dielectric if it joins two conductors,
        or a conductor and the reference."""
        conductor = list(range(self.ports))
        for g in groups or []:
            for p in g:
                conductor[p] = min(g)
        is_a = lambda i: i < self.ports
        metal = [(is_a(x) and is_a(y) and conductor[x] == conductor[y]) or not (is_a(x) or is_a(y))
                 for x, y in zip(self.pair_a[self.n_rlc:].tolist(), self.pair_b[self.n_rlc:].tolist())]
        return torch.tensor(metal, dtype=torch.bool, device=DEVICE)

    def _init_parameters(self, groups):
        """RLC edges are drawn as (f_0, Q, z) and converted to (L, C, R); RL edges as (R, f_L) (Eqs. 6 and 7).
        Random numbers are drawn in this order: f_0, z, Q (RLC edges), then R, f_L (RL edges)."""
        n = self.n_rlc
        lo, hi = math.log10(self.f_min), math.log10(self.f_max)
        log_f0 = lo + (hi - lo) * torch.rand(n, dtype=DTYPE, device=DEVICE)
        log_z = torch.randn(n, dtype=DTYPE, device=DEVICE) * LOG_Z_STD + LOG_Z_MEAN
        q_lo, q_hi = math.log10(Q_INIT_RANGE[0]), math.log10(min(Q_CAP, Q_INIT_RANGE[1]))
        q = 10 ** (q_lo + (q_hi - q_lo) * torch.rand(n, dtype=DTYPE, device=DEVICE))
        log_w0 = log_f0 + math.log10(2 * math.pi)
        self.rlc_l = torch.nn.Parameter(log_z - log_w0)             # L = z / w_0
        self.rlc_c = torch.nn.Parameter(-log_z - log_w0)            # C = 1 / (z w_0)
        self.rlc_r = torch.nn.Parameter(log_z - torch.log10(q))     # R = z / Q

        log_r = torch.randn(self.n_rl, dtype=DTYPE, device=DEVICE) + torch.where(
            self.rl_starts_as_metal(groups), LOG_R_RL_CONNECTED, LOG_R_RL_ISOLATED).to(DTYPE)
        f_lo, f_hi = LOG_FL_RANGE
        f_l = 10 ** (f_lo + (f_hi - f_lo) * torch.rand(self.n_rl, dtype=DTYPE, device=DEVICE))
        self.rl_r = torch.nn.Parameter(log_r)
        self.rl_l = torch.nn.Parameter(log_r - torch.log10(2 * torch.pi * f_l))     # L = R / (2 pi f_L)
        with torch.no_grad():
            self.project_(dict(self.named_parameters()))

    # ------------------------------------------------------------------------------- allowed range of the values

    def project_(self, P):
        """In place: clamp the parameters in P (name -> tensor, possibly with a leading batch dimension) into the allowed
        range of Eqs. 3-5."""
        l, c, r = P['rlc_l'], P['rlc_c'], P['rlc_r']
        l.copy_(l.clamp(min=LOG_L_MIN, max=LOG_L_MAX))
        c.copy_(c.clamp(min=LOG_C_MIN, max=LOG_C_MAX))
        r_lo = torch.clamp((l - c) / 2 - LOG_Q_CAP, min=LOG_R_MIN)       # R >= max(R_min, z / Q_cap)
        r.copy_(torch.maximum(r.clamp(max=LOG_R_MAX), r_lo))
        rr, rl = P['rl_r'], P['rl_l']
        rr.copy_(rr.clamp(min=LOG_R_MIN, max=LOG_R_MAX))
        l_lo = rr - LOG_2PI_FDC                                          # L >= R / (2 pi f_DCcap)
        rl.copy_(torch.minimum(torch.maximum(rl, l_lo), torch.full_like(rl, LOG_L_MAX)).clamp(min=LOG_L_MIN))

    # ---------------------------------------------------------------------------------------- element values

    def rlc_terms(self):
        """(w_0, z, Q) of every RLC edge, with the clamps of Eq. 3: w_0 = 1 / sqrt(L C), z = sqrt(L / C), Q = z / R.
        The lower bound of R depends on L and C: it is treated as a constant when differentiating."""
        l = torch.clamp(self.rlc_l, max=LOG_L_MAX)
        c = torch.clamp(self.rlc_c, max=LOG_C_MAX)
        log_z = (l - c) / 2
        r = torch.clamp(self.rlc_r, max=LOG_R_MAX)
        r = torch.clamp(r, min=torch.clamp(log_z - LOG_Q_CAP, min=LOG_R_MIN).detach())
        return 10 ** (-(l + c) / 2), 10 ** log_z, 10 ** (log_z - r)

    def rl_values(self):
        """(R, L) of every RL edge, with the clamps of Eq. 3 (the lower bound of L depends on R: constant when
        differentiating)."""
        r = torch.clamp(self.rl_r, min=LOG_R_MIN, max=LOG_R_MAX)
        l = torch.clamp(self.rl_l, min=(r - LOG_2PI_FDC).detach())
        l = torch.clamp(l, max=LOG_L_MAX)
        return 10 ** r, 10 ** l

    def rlc_elements(self):
        """(R, L, C) of every RLC edge."""
        w0, z, Q = self.rlc_terms()
        return z / Q, z / w0, 1 / (z * w0)

    def rlc_admittances(self, omega):
        """(N_freq, n_rlc) admittances 1 / Z_n of the RLC edges, written as
        j w / (z w_0 [(1 - (w / w_0)^2) + j (w / w_0) / Q]) = 1 / (R + j w L + 1 / (j w C)), which is finite at w = 0.
        omega: (N_freq, 1). Removed edges have admittance 0."""
        w0, z, Q = self.rlc_terms()
        ratio = omega / w0
        Y = cplx(torch.zeros_like(omega), omega) / cplx(z * w0 * (1 - ratio ** 2), z * w0 * ratio / Q)
        return Y * self.rlc_alive.to(DTYPE)

    def rl_admittances(self, omega):
        """(N_freq, n_rl) admittances 1 / (R + j w L) of the RL edges (0 for removed edges)."""
        R, L = self.rl_values()
        return 1 / cplx(R.expand_as(omega * R), omega * L) * self.rl_alive.to(DTYPE)

    # ------------------------------------------------------------------------------------------- forward model

    def forward(self, freq, z0=50.0, rlc_on=True):
        """S matrix (N_freq, P, P) of the netlist at the frequencies freq (Hz), by nodal analysis: every port k is
        driven in turn by the current 2 / sqrt(z0) into PkA and out of PkB (all ports at once, one multi-right-hand-side
        solve), every port is terminated in z0. rlc_on=False: the RLC edges are disconnected (stage i of the training)."""
        n_f = freq.shape[0]
        omega = 2 * torch.pi * freq.to(DTYPE).view(-1, 1)
        if rlc_on:
            Y_rlc = self.rlc_admittances(omega)
        else:
            Y_rlc = torch.zeros(n_f, self.n_rlc, dtype=CDTYPE, device=DEVICE)
        Y_edge = torch.cat([Y_rlc, self.rl_admittances(omega)], dim=1)

        # nodal admittance matrix of all nodes, including the port terminations Z0 between PpA and PpB
        n = self.n_nodes
        y0 = torch.full((n_f, self.ports), 1.0 / z0, dtype=CDTYPE, device=DEVICE)
        vals = torch.cat([Y_edge.repeat(1, 4) * self.stamp_sign, y0, y0, -y0, -y0], dim=1)
        G = torch.zeros(n_f, n * n, dtype=CDTYPE, device=DEVICE).index_add(1, self.stamp_idx, vals).view(n_f, n, n)

        # the reference node is at 0 V: its row and column are dropped. With the RLC edges off the internal nodes are
        # isolated: solve for the terminals only.
        solved = self.solved_nodes if rlc_on else self.solved_terminals
        G = G[:, solved][:, :, solved] + torch.eye(len(solved), dtype=CDTYPE, device=DEVICE) * (1.0 / R_EPS)
        sqrt_z0 = z0 ** 0.5
        pa, pb = self.port_a, self.port_b
        incidence = (torch.nn.functional.one_hot(pa, n) - torch.nn.functional.one_hot(pb, n)).T.to(CDTYPE)
        currents = (incidence * (2 / sqrt_z0)).expand(n_f, n, self.ports)[:, solved]
        v_solved = lin_solve(G, currents)
        v = torch.zeros(n_f, n, self.ports, dtype=CDTYPE, device=DEVICE).index_copy(1, solved, v_solved)

        v_ports = v[:, pa, :] - v[:, pb, :]                         # (N_freq, port m, driven port k)
        i_ports = torch.eye(self.ports, dtype=CDTYPE, device=DEVICE) * (2 / sqrt_z0) - v_ports / z0
        a_waves = (v_ports + z0 * i_ports) / (2 * sqrt_z0)          # incident waves
        b_waves = (v_ports - z0 * i_ports) / (2 * sqrt_z0)          # reflected waves
        return b_waves / torch.diagonal(a_waves, dim1=-2, dim2=-1)[:, None, :]

    # ------------------------------------------------------------------------------------- edges and SPICE export

    def node_name(self, i):
        i = int(i)
        if i < self.ports:
            return f'P{i + 1}A'
        if i < self.n_terminals:
            return f'P{i - self.ports + 1}B'
        return f'I{i - self.n_terminals}'

    def edges(self):
        """The edges still present as (kind, index, node a, node b), kind 'rlc' or 'rl', in node names."""
        out = []
        for kind, start, alive in (('rlc', 0, self.rlc_alive), ('rl', self.n_rlc, self.rl_alive)):
            for e, ok in enumerate(alive.cpu().tolist()):
                if ok:
                    out.append((kind, e, self.node_name(self.pair_a[start + e]), self.node_name(self.pair_b[start + e])))
        return out

    def edge_counts(self):
        """(RLC edges, RL edges, R_eps resistors) of the exported netlist: R_eps joins every terminal and every node
        carrying an edge to the reference."""
        edges = self.edges()
        nodes = {x for _, _, a, b in edges for x in (a, b)} | {self.node_name(t) for t in range(self.n_terminals)}
        return (sum(k == 'rlc' for k, *_ in edges), sum(k == 'rl' for k, *_ in edges), len(nodes - {'P1B'}))

    @torch.no_grad()
    def write_spice(self, path, name='fitted', z0=50.0):
        """SPICE subcircuit with pins P1A P1B ... PPA PPB (port p: PpA positive, PpB negative; P1B is the reference
        node). Edges still present only: Rrlc/Lrlc/Crlc in series for an RLC edge, Rrl/Lrl for an RL edge, plus the
        R_eps resistors."""
        fmt = lambda v: f'{float(v):.12g}'
        R, L, C = (t.cpu() for t in self.rlc_elements())
        Rl, Ll = (t.cpu() for t in self.rl_values())
        lines = []
        for kind, e, a, b in self.edges():
            if kind == 'rlc':
                lines += [f'* RLC[{e}] {a}-{b}', f'Rrlc{e} {a} nrlc{e}_1 {fmt(R[e])}',
                          f'Lrlc{e} nrlc{e}_1 nrlc{e}_2 {fmt(L[e])}', f'Crlc{e} nrlc{e}_2 {b} {fmt(C[e])}']
            else:
                lines += [f'* RL[{e}] {a}-{b}', f'Rrl{e} {a} nrl{e}_1 {fmt(Rl[e])}', f'Lrl{e} nrl{e}_1 {b} {fmt(Ll[e])}']
        nodes = {x for _, _, a, b in self.edges() for x in (a, b)} | {self.node_name(t) for t in range(self.n_terminals)}
        eps = [f'Reps_{m} {m} P1B {fmt(R_EPS)}' for m in sorted(nodes - {'P1B'})]
        pins = [self.node_name(x) for p in range(self.ports) for x in (p, p + self.ports)]
        header = [f'* {name}: {self.ports}-port fitted model, reference z0 = {z0:g} Ohm',
                  '* port p is PpA (+) to PpB (-); the reference node is P1B',
                  f'.subckt {name} ' + ' '.join(pins)]
        with open(path, 'w') as fh:
            fh.write('\n'.join(header + ['* ---- edges'] + lines + ['* ---- R_eps'] + eps + [f'.ends {name}', '']))


def load_netlist(path):
    """(Netlist, checkpoint dict) from a file written by train.py."""
    ck = torch.load(path, map_location=DEVICE)
    net = Netlist(**ck['config'])
    net.load_state_dict(ck['state_dict'])
    return net, ck
