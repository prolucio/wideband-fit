"""Touchstone -> matrix file.

Parse an N-port Touchstone (v1) file, e.g. an HFSS export, and store its S, Z or Y matrices as a compressed numpy archive
(``.npz``) with the fields:

    freq    (n_freq,)                real, Hz
    params  (n_freq, n_port, n_port) complex   the network matrix per frequency
    kind    ()                       str        'S', 'Z' or 'Y'
    z0      ()                       float      reference impedance, ohm

The frequency points are taken verbatim from the file (HFSS "20pt/dec" sweeps
are log-spaced); no resampling is done here.

Grounded targets (every terminal a port against the solver's reference, e.g. a Q3D 8-port of a 4-port floating
structure): --ports reorders the ports to A1..An, B1..Bn (port p of the floating structure is the pair Ap, Bp) and
--grounded marks the file (field ``grounded`` = True); train.py then fits Circuit(grounded=True).
    python parse_touchstone.py q3d.s8p -o data/model_lf8.npz --ports 7,4,8,5,2,1,3,6 --grounded

Usage
-----
    python parse_touchstone.py model.s4p                    # -> model_matrix.npz
    python parse_touchstone.py model.s4p -o data/model.npz
    python parse_touchstone.py model.s4p -o data/model_hf.npz --fmin 1e5 --fmax 1e9   # keep 100 kHz .. 1 GHz

train.py only needs the S file (it derives Z and Y itself). For inspection, --derive-y / --derive-z / --derive-all
also write the normalised matrices Z/z0 and Y*z0 computed from S (same freq and z0), next to the S file:
``S_parameters`` in the name is replaced by ``Y_parameters`` / ``Z_parameters`` (otherwise ``_Y`` / ``_Z`` is inserted
before ``_matrix``). Use -oy / -oz to choose the paths.
    Z/z0 = (I + S)(I - S)^-1,   Y*z0 = (I - S)(I + S)^-1
These are only accurate when (I -/+ S) is well conditioned; a warning gives the worst condition number.

--z0 R re-normalizes the matrices to a new real reference impedance R before
saving (z0 in the npz becomes R; derived Y/Z follow it). S is re-referenced
(real z0), Z/z0 and Y*z0 are rescaled so the physical Z and Y are unchanged.
The starting point is the file's own z0 (R field, else 50).
"""

import argparse
import os
import re

import numpy as np

FREQ_MULT = {"hz": 1.0, "khz": 1e3, "mhz": 1e6, "ghz": 1e9, "thz": 1e12}


def parse_touchstone(path):
    """Return (freq_hz, params, kind, z0) for an N-port Touchstone file."""
    n_ports = 2
    m = re.search(r"\.s(\d+)p$", path, re.IGNORECASE)
    if m:
        n_ports = int(m.group(1))

    freq_unit, kind, fmt, z0 = "ghz", "s", "ma", 50.0
    tokens = []
    with open(path) as fh:
        for raw in fh:
            line = raw.split("!", 1)[0].strip()  # drop inline/whole-line comments
            if not line:
                continue
            if line.startswith("#"):
                opt = line[1:].lower().split()
                if opt:
                    freq_unit = opt[0]
                if len(opt) > 1:
                    kind = opt[1]
                if len(opt) > 2:
                    fmt = opt[2]
                if "r" in opt:
                    z0 = float(opt[opt.index("r") + 1])
                continue
            tokens.extend(float(t) for t in line.replace(",", " ").split())

    kind = kind.upper()
    if kind not in ("S", "Z", "Y"):
        raise ValueError(f"unsupported parameter type {kind!r} (need S, Z or Y)")

    rec_len = 1 + 2 * n_ports * n_ports
    if len(tokens) % rec_len:
        raise ValueError(
            f"token count {len(tokens)} is not a multiple of the record length "
            f"{rec_len} for a {n_ports}-port {fmt.upper()} file"
        )

    data = np.asarray(tokens, dtype=float).reshape(-1, rec_len)
    freq = data[:, 0] * FREQ_MULT[freq_unit]
    a = data[:, 1::2]
    b = data[:, 2::2]

    if fmt == "ri":
        vals = a + 1j * b
    elif fmt == "ma":
        vals = a * np.exp(1j * np.deg2rad(b))
    elif fmt == "db":
        vals = 10.0 ** (a / 20.0) * np.exp(1j * np.deg2rad(b))
    else:
        raise ValueError(f"unsupported data format {fmt!r} (need RI, MA or DB)")

    n_freq = vals.shape[0]
    if n_ports == 2:
        # Touchstone 2-port ordering is 11, 21, 12, 22 (the historical exception).
        params = np.empty((n_freq, 2, 2), dtype=complex)
        params[:, 0, 0] = vals[:, 0]
        params[:, 1, 0] = vals[:, 1]
        params[:, 0, 1] = vals[:, 2]
        params[:, 1, 1] = vals[:, 3]
    else:
        params = vals.reshape(n_freq, n_ports, n_ports)  # row-major

    return freq, params, kind, z0


def renormalize(params, kind, z0, z0_new):
    """Re-reference a matrix file from z0 to z0_new (real impedances). S is
    re-referenced; Z/z0 and Y*z0 are rescaled so physical Z, Y are unchanged."""
    if z0_new == z0:
        return params
    if kind == "S":
        g = (z0_new - z0) / (z0_new + z0)
        eye = np.eye(params.shape[-1])
        return (params - g * eye) @ np.linalg.inv(eye - g * params)
    return params * (z0 / z0_new if kind == "Z" else z0_new / z0)


def s_to_zy(s):
    """Return (Z/z0, Y*z0, worst_cond) from S (n_freq, n, n): the normalised matrices."""
    n = s.shape[-1]
    eye = np.eye(n)
    a, b = eye + s, eye - s
    # Zn = a b^-1  ->  Zn^T = b^-T a^T, solved per frequency
    z = np.swapaxes(np.linalg.solve(np.swapaxes(b, 1, 2), np.swapaxes(a, 1, 2)), 1, 2)
    y = np.swapaxes(np.linalg.solve(np.swapaxes(a, 1, 2), np.swapaxes(b, 1, 2)), 1, 2) 
    cond = max(np.linalg.cond(b).max(), np.linalg.cond(a).max())
    return z, y, cond


def derived_path(path, kind):
    base = os.path.splitext(path)[0]
    if "S_parameters" in base:
        return base.replace("S_parameters", f"{kind}_parameters") + ("" if base.endswith("_matrix") else "_matrix") + ".npz"
    return (base[:-7] + f"_{kind}_matrix" if base.endswith("_matrix") else base + f"_{kind}_matrix") + ".npz"


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("input", help="Touchstone file (.s<N>p)")
    ap.add_argument("-o", "--output", help="output .npz (default: <input>_matrix.npz)")
    ap.add_argument("--derive-y", action="store_true", help="also write Y computed from S")
    ap.add_argument("--derive-z", action="store_true", help="also write Z computed from S")
    ap.add_argument("--derive-all", action="store_true", help="--derive-y and --derive-z")
    ap.add_argument("-oy", "--output-y", help="Y output .npz (default: derived from the S name)")
    ap.add_argument("-oz", "--output-z", help="Z output .npz (default: derived from the S name)")
    ap.add_argument("--fmin", type=float, default=None, help="keep the points f >= this (Hz); default: all")
    ap.add_argument("--fmax", type=float, default=None, help="keep the points f <= this (Hz); default: all")
    ap.add_argument("--ports", help="new port order, 1-based, comma separated (port i of the output is this "
                                    "list's i-th port of the file); default: as in the file")
    ap.add_argument("--grounded", action="store_true",
                    help="mark the file as grounded: ports A1..An, B1..Bn, each against the reference (after --ports)")
    ap.add_argument("--z0", type=float, metavar="R",
                    help="re-normalize to this real reference impedance (ohm) before saving")
    args = ap.parse_args()
    if args.z0 is not None and args.z0 <= 0:
        ap.error("--z0 must be positive")
    want_y = args.derive_y or args.derive_all or args.output_y
    want_z = args.derive_z or args.derive_all or args.output_z

    out = args.output or os.path.splitext(args.input)[0] + "_matrix.npz"
    freq, params, kind, z0 = parse_touchstone(args.input)
    keep = np.ones(len(freq), bool)
    if args.fmin is not None:
        keep &= freq >= args.fmin * (1 - 1e-9)
    if args.fmax is not None:
        keep &= freq <= args.fmax * (1 + 1e-9)
    if not keep.all():
        print(f"band cut: kept {keep.sum()} of {len(freq)} points")
        freq, params = freq[keep], params[keep]
    if args.ports:
        order = [int(x) - 1 for x in args.ports.split(",")]
        if sorted(order) != list(range(params.shape[1])):
            ap.error(f"--ports must list every port 1..{params.shape[1]} once")
        params = params[:, order][:, :, order]
        print(f"ports reordered: {args.ports}")
    if args.grounded and params.shape[1] % 2:
        ap.error("--grounded needs an even number of ports (A1..An, B1..Bn)")
    if args.z0 is not None and args.z0 != z0:
        print(f"re-normalizing {kind} from z0={z0:g} to z0={args.z0:g} ohm")
        params = renormalize(params, kind, z0, args.z0)
        z0 = args.z0
    np.savez_compressed(out, freq=freq, params=params, kind=kind, z0=z0, grounded=args.grounded)

    print(f"{args.input}: {kind}-parameters, z0={z0:g} ohm, "
          f"{params.shape[1]}-port, {len(freq)} points "
          f"{freq[0]:g}-{freq[-1]:g} Hz{', grounded' if args.grounded else ''}")
    print(f"wrote {out}")

    if want_y or want_z:
        if kind != "S":
            ap.error(f"--derive-* needs an S-parameter file, got {kind}")
        z, y, cond = s_to_zy(params)
        if cond > 1e8:
            print(f"warning: (I-S)/(I+S) condition number up to {cond:.2e}; derived Y/Z "
                  f"may be inaccurate (worst near DC)")
        for want, mat, k, given in ((want_y, y, "Y", args.output_y), (want_z, z, "Z", args.output_z)):
            if want:
                p = given or derived_path(out, k)
                np.savez_compressed(p, freq=freq, params=mat, kind=k, z0=z0)
                print(f"wrote {p} ({k}-parameters derived from S)")


if __name__ == "__main__":
    main()
