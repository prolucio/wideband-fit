"""Transient simulation of exported netlists (Fig. 4 of the paper).

    python transient.py runs/lf/100-131/seed_118.cir runs/hf/.../seed_272.cir --labels LF HF --out runs/transient

Test bench: a square wave on port 1 (ideal voltage source between P1A and the reference P1B), an RC load on port 2 (R_LOAD in
parallel with C_LOAD between P2A and P2B), the other ports open. Writes
  <out>_<label>.cir   SPICE deck of the bench (SPICE-compatible simulators, e.g. ngspice or LTspice: `.tran`), to run the same
                      test in a real SPICE;
  <out>.csv, <out>.png  the input and the output voltage V(P2A) - V(P2B) of every netlist, from the built-in integrator.
The built-in integrator solves the modified nodal equations C x' + G x = b u(t) of the same circuit with a fixed step DT and
BDF2 (L-stable). The trapezoidal rule, SPICE's default, is not used: at a fixed step it diverges on these stiff netlists
(poles from about 1e5 to 1e12 rad/s and resistors up to 1e12 Ohm).
"""
import argparse
import re

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

# Test bench
V_HIGH = 10.0         # V, square wave between 0 and V_HIGH
PERIOD = 1e-6         # s
DUTY = 0.5
RISE = FALL = 1e-9    # s
R_LOAD, C_LOAD = 10.0, 1e-9
T_STOP, DT = 3e-6, 0.1e-9


def read_netlist(path):
    """([(name, node a, node b, value)] of the R, L, C elements, pins of the subcircuit, text of the subcircuit)."""
    text = open(path).read()
    elements, pins = [], None
    for line in text.splitlines():
        t = line.split()
        if not t or t[0].startswith('*'):
            continue
        if t[0].lower() == '.subckt':
            pins = t[2:]
        elif t[0][0] in 'RLC':
            elements.append((t[0], t[1], t[2], float(t[3])))
    return elements, pins, text


def square_wave(t):
    """The source voltage at times t (numpy): low, rises over RISE at the start of each period, high for DUTY of it, falls over FALL."""
    ph = np.mod(t, PERIOD)
    return V_HIGH * np.clip(ph / RISE, 0, 1) * np.clip((DUTY * PERIOD + FALL - ph) / FALL, 0, 1)


def write_deck(path, netlist_path, name='fitted'):
    """SPICE deck of the test bench (the subcircuit is copied into it)."""
    _, pins, text = read_netlist(netlist_path)
    nodes = {'P1A': 'in', 'P1B': '0', 'P2A': 'out', 'P2B': 'outb'}
    with open(path, 'w') as fh:
        fh.write(f'* transient test bench of {netlist_path}\n{text}\n')
        fh.write(f'Vin in 0 PULSE(0 {V_HIGH:g} 0 {RISE:g} {FALL:g} {DUTY * PERIOD - RISE:g} {PERIOD:g})\n')
        fh.write(f'X1 {" ".join(nodes.get(p, p.lower()) for p in pins)} {name}\n')
        fh.write(f'Rload out outb {R_LOAD:g}\nCload out outb {C_LOAD:g}\n.tran {DT:g} {T_STOP:g}\n.end\n')


def build_system(netlist_path):
    """Matrices G, C and the input vector b of C x' + G x = b u(t); x = voltages of all nodes but the reference, currents of the
    inductors and of the source. Returns also the function that reads V(P2A) - V(P2B) from x."""
    elements, pins, _ = read_netlist(netlist_path)
    elements += [('Rload', 'P2A', 'P2B', R_LOAD), ('Cload', 'P2A', 'P2B', C_LOAD)]
    ref = 'P1B'
    nodes = sorted({n for _, a, b, _ in elements for n in (a, b)} | set(pins))
    nodes = [n for n in nodes if n != ref]
    index = {n: i for i, n in enumerate(nodes)}
    inductors = [e for e in elements if e[0][0] == 'L']
    size = len(nodes) + len(inductors) + 1
    G, C, b = np.zeros((size, size)), np.zeros((size, size)), np.zeros(size)

    def stamp(M, a, c, v):
        for x in (a, c):
            if x != ref:
                M[index[x], index[x]] += v
        if a != ref and c != ref:
            M[index[a], index[c]] -= v
            M[index[c], index[a]] -= v
    k = len(nodes)
    for name, a, c, v in elements:
        if name[0] == 'R':
            stamp(G, a, c, 1 / v)
        elif name[0] == 'C':
            stamp(C, a, c, v)
        else:                                   # inductor: current k leaves a, enters c; -(Va - Vc) + L dI/dt = 0
            for x, sign in ((a, 1), (c, -1)):
                if x != ref:
                    G[index[x], k] += sign
                    G[k, index[x]] -= sign
            C[k, k] = v
            k += 1
    G[index['P1A'], k] += 1                     # source current leaves P1A; V(P1A) = u(t)
    G[k, index['P1A']] += 1
    b[k] = 1.0
    output = lambda x: x[index['P2A']] - x[index['P2B']]
    return G, C, b, output


def simulate(G, C, b):
    """x(t) at t = 0, DT, ..., T_STOP from the zero state (the source is 0 V at t = 0)."""
    n = round(T_STOP / DT)
    t = np.arange(n + 1) * DT
    u = square_wave(t)
    lu = lambda A: torch.linalg.lu_factor(torch.from_numpy(np.ascontiguousarray(A)))
    solve = lambda f, rhs: torch.linalg.lu_solve(*f, torch.from_numpy(rhs)[:, None]).numpy()[:, 0]
    x = np.zeros((n + 1, len(b)))
    # BDF2, the first step backward Euler
    x[1] = solve(lu(C / DT + G), C @ x[0] / DT + b * u[1])
    f = lu(1.5 * C / DT + G)
    for i in range(1, n):
        x[i + 1] = solve(f, C @ (2 * x[i] - 0.5 * x[i - 1]) / DT + b * u[i + 1])
    return t, u, x


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('netlists', nargs='+')
    ap.add_argument('--labels', nargs='+', help='one label per netlist (default: the file names)')
    ap.add_argument('--out', default='runs/transient')
    ap.add_argument('--window', type=float, nargs=2, metavar=('T0', 'T1'), help='time range of the plot (us)')
    args = ap.parse_args()
    labels = args.labels or [re.sub(r'\.cir$', '', p.split('/')[-1]) for p in args.netlists]
    traces = {}
    for path, label in zip(args.netlists, labels):
        write_deck(f'{args.out}_{label}.cir', path)
        G, C, b, output = build_system(path)
        t, u, x = simulate(G, C, b)
        traces[label] = np.array([output(row) for row in x])
        print(f'{label}: {len(b)} unknowns, {len(t) - 1} steps of {DT * 1e9:g} ns, '
              f'output range {traces[label].min():.3f} .. {traces[label].max():.3f} V')
    np.savetxt(f'{args.out}.csv', np.column_stack([t * 1e6, u, *traces.values()]), delimiter=',',
               header=','.join(['time_us', 'input_V'] + [f'output_{k}_V' for k in traces]), comments='')
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(t * 1e6, u, 'k', lw=1, label='input')
    for label, v in traces.items():
        ax.plot(t * 1e6, v, lw=1, label=f'output {label}')
    ax.set_xlabel('time (µs)')
    ax.set_ylabel('voltage (V)')
    if args.window:
        ax.set_xlim(*args.window)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(f'{args.out}.png', dpi=150)


if __name__ == '__main__':
    main()
