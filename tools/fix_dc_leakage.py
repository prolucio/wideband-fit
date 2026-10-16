"""Remove a field solver's spurious DC conductance from an S-parameter set.

    python tools/fix_dc_leakage.py IN.s4p --groups 1,2 3,4 -o OUT.npz [--touchstone OUT.s4p]

At DC an isolated conductor exchanges current only through its own ports: for the ports of one conductor (a group),
the currents sum to zero for any port voltages, i.e. the rows (and columns) of the DC admittance matrix Y(0) of each
group sum to zero. The part of Y(0) that violates this is

    Y_leak = Y(0) - P Y(0) P,    P = I - N (N^T N)^-1 N^T,

where N has one indicator column per group (1 on the ports of that conductor) and P removes from a port vector the
part that raises a whole conductor (the mean over its ports). Y_leak (made symmetric) is subtracted from Y(f) at every
frequency f, and S is rebuilt from the corrected Y. Assumption: the spurious conductance does not depend on frequency.

Every port must belong to exactly one group, and the file must contain a DC point. When every port is referred to one
common conductor (e.g. a ground plate), group the ports by the conductor of their other terminal: two traces over a
plate, trace A at ports 1, 2 and trace B at ports 3, 4, give --groups 1,2 3,4.
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from parse_touchstone import parse_touchstone  # noqa: E402


def s_to_y(S, z0):
    I = np.eye(S.shape[-1])
    return np.linalg.solve((I + S).transpose(0, 2, 1), (I - S).transpose(0, 2, 1)).transpose(0, 2, 1) / z0


def y_to_s(Y, z0):
    I = np.eye(Y.shape[-1])
    return np.linalg.solve((I + z0 * Y).transpose(0, 2, 1), (I - z0 * Y).transpose(0, 2, 1)).transpose(0, 2, 1)


def group_matrices(n, groups):
    """(P, N): N (n, n_groups) indicator columns, P the projector removing each group's mean."""
    N = np.zeros((n, len(groups)))
    for k, g in enumerate(groups):
        N[list(g), k] = 1.0
    if not np.array_equal(N.sum(1), np.ones(n)):
        raise ValueError(f'every port must be in exactly one group: {groups}')
    return np.eye(n) - N @ np.linalg.solve(N.T @ N, N.T), N


def remove_dc_leakage(freq, S, z0, groups):
    """Return (S_fixed, Y_leak). freq (F,) Hz with freq[0] == 0, S (F, n, n), groups: lists of 0-based ports."""
    if freq[0] != 0:
        raise ValueError('the first frequency must be DC (0 Hz)')
    P, _ = group_matrices(S.shape[-1], groups)
    Y = s_to_y(S, z0)
    Y0 = Y[0].real
    Y_leak = Y0 - P @ Y0 @ P
    Y_leak = 0.5 * (Y_leak + Y_leak.T)
    S_fixed = y_to_s(Y - Y_leak, z0)
    S_fixed[0] = S_fixed[0].real
    return S_fixed, Y_leak


def write_touchstone(path, freq, S, z0, comments=()):
    """Touchstone v1, '# GHz S MA R z0', one matrix row per line, 17 significant digits."""
    with open(path, 'w') as fh:
        for c in comments:
            fh.write(f'! {c}\n')
        fh.write(f'# GHz S MA R {z0:.6f}\n')
        for f, M in zip(freq, S):
            for i in range(M.shape[0]):
                pairs = ' '.join(f'{abs(x):.17g} {np.degrees(np.angle(x)):.17g}' for x in M[i])
                fh.write((f'{f / 1e9:.17g} ' if i == 0 else ' ' * 4) + pairs + '\n')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Touchstone .s<N>p or .npz (freq, params, kind S, z0)')
    ap.add_argument('--groups', nargs='+', required=True, help='ports of each isolated conductor, 1-based: 1,2 3,4')
    ap.add_argument('-o', '--output', required=True, help='output .npz (freq, params, kind, z0)')
    ap.add_argument('--touchstone', help='also write this Touchstone file')
    a = ap.parse_args()
    if a.input.endswith('.npz'):
        d = np.load(a.input)
        freq, S, kind, z0 = d['freq'], d['params'], str(d['kind']), float(d['z0'])
    else:
        freq, S, kind, z0 = parse_touchstone(a.input)
    if kind.upper() != 'S':
        raise ValueError(f'expected S-parameters, got {kind}')
    groups = [[int(p) - 1 for p in g.split(',')] for g in a.groups]
    S_fixed, Y_leak = remove_dc_leakage(freq, S, z0, groups)
    _, N = group_matrices(S.shape[-1], groups)
    np.set_printoptions(precision=6, linewidth=160)
    print(f'{a.input}: {S.shape[-1]} ports, {len(freq)} points, groups {a.groups}')
    print('removed conductance per conductor, N^T Y_leak N (uS; diagonal: conductor to the rest, off-diagonal: minus '
          'the conductance between conductors):\n', N.T @ Y_leak @ N * 1e6)
    np.savez_compressed(a.output, freq=freq, params=S_fixed, kind=np.array('S'), z0=np.array(z0))
    print(f'wrote {a.output}')
    if a.touchstone:
        write_touchstone(a.touchstone, freq, S_fixed, z0, [f'{os.path.basename(a.input)} with the spurious DC '
                                                           f'conductance removed (tools/fix_dc_leakage.py)'])
        print(f'wrote {a.touchstone}')


if __name__ == '__main__':
    main()
