"""Synthetic test targets with low-frequency resonances: a measured response plus passive series-RLC branches added to
its admittance matrix (Y_new = Y + branch stamps), so the result is exactly passive and reciprocal, on the same frequency
grid (a series RLC has zero admittance at DC).
    python tools/make_synth_target.py SOURCE_S.npz [--out-dir DIR]   -> DIR/<name>/{S,Z,Y}_parameters_<name>_matrix.npz
Branches: (port a, port b (0 = ground), f0 Hz, Q, L H); C = 1/(w0^2 L), R = w0 L / Q."""
import argparse
import os
import numpy as np
TARGETS = {
    # inside the W1 window (L <= 1 uH, C <= 10 nF: lowest single-branch f0 = 1.6 MHz) and inside the 1.05e7 Hz pretrain window
    'res_mhz': [(1, 3, 5e6, 10, 1e-6), (4, 0, 2e6, 20, 1e-6)],
    # needs a wider window (train.py --l-max 1e-3 --c-max 1e-5); resonances in the low band (<= 1e5 Hz)
    'res_khz': [(1, 3, 2e4, 10, 1e-4), (4, 0, 5e4, 20, 5e-5)],
}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('source', help='S-parameter .npz of the base response')
    ap.add_argument('--out-dir', default='synth_targets')
    args = ap.parse_args()
    d = np.load(args.source)
    f, S, z0 = d['freq'], d['params'], float(d['z0'])
    n = S.shape[-1]
    I = np.eye(n)
    Y = np.linalg.solve(I + S, I - S) / z0          # Y (S), from Yn = (I+S)^-1 (I-S)
    w = 2 * np.pi * f
    for name, br in TARGETS.items():
        Yn = Y.copy()
        lines = []
        for a, b, f0, Q, L in br:
            w0 = 2 * np.pi * f0
            C, R = 1 / (w0 ** 2 * L), w0 * L / Q
            with np.errstate(divide='ignore'):
                zb = R + 1j * w * L + np.where(w > 0, 1 / (1j * w * C + 1e-300), np.inf)
            y = np.where(w > 0, 1 / zb, 0)
            Yn[:, a - 1, a - 1] += y
            if b:
                Yn[:, b - 1, b - 1] += y
                Yn[:, a - 1, b - 1] -= y
                Yn[:, b - 1, a - 1] -= y
            lines.append(f'  series RLC port{a}-{"port%d" % b if b else "GND"}: f0 {f0:g} Hz, Q {Q:g}, '
                         f'R {R:.4g} Ohm, L {L:.4g} H, C {C:.4g} F')
        Ynorm = Yn * z0
        Sn = np.linalg.solve((I + Ynorm).transpose(0, 2, 1), (I - Ynorm).transpose(0, 2, 1)).transpose(0, 2, 1)
        # Z/z0 is NaN where it does not exist (I - S singular, e.g. at DC of isolated conductors)
        s_imz = np.linalg.svd(I - Sn, compute_uv=False)
        has_z = s_imz[:, -1] >= 1e-12 * s_imz[:, 0]
        Zn = np.full_like(Sn, np.nan)
        Zn[has_z] = np.linalg.solve(I - Sn[has_z], I + Sn[has_z])
        out = os.path.join(args.out_dir, name)
        os.makedirs(out, exist_ok=True)
        for kind, P in (('S', Sn), ('Z', Zn), ('Y', Ynorm)):
            np.savez(f'{out}/{kind}_parameters_{name}_matrix.npz', freq=f, params=P, kind=kind, z0=z0)
        sv = np.linalg.svd(Sn, compute_uv=False).max()
        open(f'{out}/README.txt', 'w').write(
            f'{name}: synthetic target = {args.source} + passive branches added to Y '
            f'(tools/make_synth_target.py):\n' + '\n'.join(lines) +
            f'\nmax singular value of S over the grid: {sv:.6f} (<= 1: passive)\n')
        print(name, '\n'.join(lines), f'max sigma(S) {sv:.6f}')


if __name__ == '__main__':
    main()
