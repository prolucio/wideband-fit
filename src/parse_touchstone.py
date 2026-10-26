"""Touchstone (.sNp) -> .npz target for train.py.

    python src/parse_touchstone.py model.s4p -o data/model.npz [--fmin 1e5] [--fmax 1e9]

Reads an N-port Touchstone v1 file of S-parameters (RI, MA or DB format; the frequency points are kept as they are, no
resampling) and writes an .npz with
    freq    (N_f,)          Hz
    params  (N_f, N, N)     complex S-parameters
    z0      ()              reference impedance (Ohm)
--fmin / --fmax keep only the points in that range (DC, f = 0, is kept only without --fmin).
"""
import argparse
import re

import numpy as np

FREQ_UNIT = {'hz': 1.0, 'khz': 1e3, 'mhz': 1e6, 'ghz': 1e9, 'thz': 1e12}


def parse_touchstone(path):
    """(freq in Hz, S (N_f, N, N), z0) of a Touchstone file."""
    n_ports = int(re.search(r'\.s(\d+)p$', path, re.IGNORECASE).group(1))
    unit, kind, fmt, z0 = 'ghz', 's', 'ma', 50.0       # defaults of the format
    tokens = []
    with open(path) as fh:
        for raw in fh:
            line = raw.split('!', 1)[0].strip()         # drop comments
            if not line:
                continue
            if line.startswith('#'):
                option = line[1:].lower().split()
                unit, kind, fmt = (option + [unit, kind, fmt][len(option):])[:3]   # omitted options keep the defaults
                if 'r' in option:
                    z0 = float(option[option.index('r') + 1])
                continue
            tokens.extend(float(t) for t in line.replace(',', ' ').split())
    if kind != 's':
        raise ValueError(f'{path}: S-parameters needed, got {kind.upper()}')

    record = 1 + 2 * n_ports ** 2
    if len(tokens) % record:
        raise ValueError(f'{path}: {len(tokens)} numbers is not a multiple of the record length {record}')
    data = np.asarray(tokens).reshape(-1, record)
    freq = data[:, 0] * FREQ_UNIT[unit]
    a, b = data[:, 1::2], data[:, 2::2]
    if fmt == 'ri':
        values = a + 1j * b
    elif fmt == 'ma':
        values = a * np.exp(1j * np.deg2rad(b))
    elif fmt == 'db':
        values = 10 ** (a / 20) * np.exp(1j * np.deg2rad(b))
    else:
        raise ValueError(f'{path}: unsupported data format {fmt!r}')

    S = values.reshape(-1, n_ports, n_ports)             # row-major
    if n_ports == 2:                                     # 2-port files are ordered 11, 21, 12, 22
        S = values.reshape(-1, 2, 2).transpose(0, 2, 1)
    return freq, S, z0


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('input', help='Touchstone file (.sNp)')
    ap.add_argument('-o', '--output', required=True, help='output .npz')
    ap.add_argument('--fmin', type=float, help='keep the points f >= this (Hz)')
    ap.add_argument('--fmax', type=float, help='keep the points f <= this (Hz)')
    args = ap.parse_args()
    freq, S, z0 = parse_touchstone(args.input)
    keep = np.ones(len(freq), bool)
    if args.fmin is not None:
        keep &= freq >= args.fmin * (1 - 1e-9)
    if args.fmax is not None:
        keep &= freq <= args.fmax * (1 + 1e-9)
    np.savez_compressed(args.output, freq=freq[keep], params=S[keep], z0=z0)
    print(f'{args.input}: {S.shape[1]}-port, z0 = {z0:g} Ohm, kept {keep.sum()} of {len(freq)} points '
          f'({freq[keep][0]:g}-{freq[keep][-1]:g} Hz) -> {args.output}')


if __name__ == '__main__':
    main()
