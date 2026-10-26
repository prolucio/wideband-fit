"""Errors of saved netlists against a target (Tables II and III of the paper).

    python evaluate.py --target data/hf.npz runs/hf/100-131/seed_100.pt [more checkpoints] [--plots] [--scores scores.csv]

For every checkpoint: magnitude error (dB) and phase error (degrees), averaged over S, Z, Y (metrics.py); the number of
RLC edges, RL edges and R_eps resistors; passivity on a dense frequency grid. --plots writes <checkpoint>_{S,Z,Y}.png
(target and fit) and <checkpoint>_{S,Z,Y}.csv (the plotted data, DC row included). --scores writes the results of all
the checkpoints in the format of the scores.csv of train.py (for seed_statistics.py).
"""
import argparse
import csv
import os

import torch

from metrics import fit_errors, is_passive, magnitude_error, phase_error
from netlist import load_netlist
from plotting import plot_bode, write_bode_csv
from sparams import CONVERT, DEVICE, DTYPE, load_target


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('checkpoints', nargs='+')
    ap.add_argument('--target', required=True)
    ap.add_argument('--plots', action='store_true')
    ap.add_argument('--scores', help='write the results as a scores.csv')
    args = ap.parse_args()

    target = load_target(args.target)
    freq = torch.tensor(target.freq, dtype=DTYPE, device=DEVICE)
    rows = []
    for path in args.checkpoints:
        net, ck = load_netlist(path)
        with torch.no_grad():
            S = net(freq, z0=target.z0).cpu()
        errors = fit_errors(target, S)
        n_rlc, n_rl, n_eps = net.edge_counts()
        passive = is_passive(net, target.z0, target.freq)
        rows.append(dict(seed=ck['seed'], magnitude_db=magnitude_error(errors), phase_deg=phase_error(errors),
                         rlc_edges=n_rlc, rl_edges=n_rl, eps_edges=n_eps, passive=passive))
        print(f'{path}: magnitude {magnitude_error(errors):.4f} dB, phase {phase_error(errors):.3f} deg '
              f'(S {errors["S"]["magnitude"]:.4f}, Z {errors["Z"]["magnitude"]:.4f}, Y {errors["Y"]["magnitude"]:.4f} dB); '
              f'{n_rlc} RLC + {n_rl} RL + {n_eps} R_eps edges; {"passive" if passive else "NOT PASSIVE"}')
        if args.plots:
            stem = os.path.splitext(path)[0]
            for domain, h in target.domains().items():
                h_hat = CONVERT[domain](S).numpy()
                plot_bode(target.freq, h.numpy(), h_hat, f'{stem}_{domain}.png', domain)
                write_bode_csv(target.freq, h.numpy(), h_hat, f'{stem}_{domain}.csv')
    if args.scores:
        with open(args.scores, 'w', newline='') as fh:
            writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)


if __name__ == '__main__':
    main()
