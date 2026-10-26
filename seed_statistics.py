"""Statistics of the fits over many seeds (Tables II and III of the paper).

    python seed_statistics.py --lf runs/lf/*/scores.csv --hf runs/hf/*/scores.csv [--tex]

Reads the scores.csv files (written by train.py, or by evaluate.py --scores) of the LF and of the HF target, each the seeds of
one batch, and prints
  - Table II: best (smallest), q25, median and q75 of the magnitude error, the phase error and the number of edges
    (RLC + RL + R_eps). Each column has its own best value and quartiles, independently of the seed;
  - Table III: the median and the best netlist, ranked by magnitude error only, with their phase error and edge counts, and
    the checkpoint of each (evaluate.py can plot it). The median is the lower median.
As markdown; with --tex also as the LaTeX tables of the paper.
"""
import argparse
import csv

import numpy as np


def load(paths):
    rows = []
    for path in paths:
        for row in csv.DictReader(open(path)):
            rows.append({'file': path.replace('scores.csv', f'seed_{row["seed"]}.pt'),
                         **{k: float(v) if k != 'passive' else v == 'True' for k, v in row.items()}})
    rows.sort(key=lambda r: r['magnitude_db'])
    for r in rows:
        r['edges'] = r['rlc_edges'] + r['rl_edges'] + r['eps_edges']
    return rows


def table_ii(rows):
    """[(name, decimals, (best, q25, median, q75))] of the three columns."""
    out = []
    for name, key, decimals in (('$\\overline{\\Delta|\\!\\cdot\\!|}$\\,(\\SI{}{\\decibel})', 'magnitude_db', 3),
                                ('$\\overline{\\Delta\\angle\\!\\cdot}$\\,($^\\circ$)', 'phase_deg', 3),
                                ('\\#\\,edges', 'edges', 0)):
        v = [r[key] for r in rows]
        q25, med, q75 = np.percentile(v, [25, 50, 75])
        out.append((name, decimals, (min(v), q25, med, q75)))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--lf', nargs='+', required=True)
    ap.add_argument('--hf', nargs='+', required=True)
    ap.add_argument('--tex', action='store_true', help='also print the LaTeX tables')
    args = ap.parse_args()
    data = {'LF': load(args.lf), 'HF': load(args.hf)}
    for target, rows in data.items():
        print(f'{target}: {len(rows)} seeds, {sum(r["passive"] for r in rows)} passive')

    labels = ['magnitude (dB)', 'phase (deg)', 'edges']
    print('\nTable II\n\n| target | | best | q25 | median | q75 |\n|---|---|---|---|---|---|')
    for target, rows in data.items():
        for label, (_, decimals, values) in zip(labels, table_ii(rows)):
            print(f'| {target} | {label} | ' + ' | '.join(f'{v:.{decimals}f}' for v in values) + ' |')

    picks = [(f'{target} ({which})', rows[(len(rows) - 1) // 2] if which == 'median' else rows[0])
             for target, rows in data.items() for which in ('median', 'best')]
    print('\nTable III (ranked by magnitude error)\n\n| target | magnitude (dB) | phase (deg) | RLC | RL | R_eps | edges | checkpoint |'
          '\n|---|---|---|---|---|---|---|---|')
    for name, r in picks:
        print(f'| {name}, seed {int(r["seed"])} | {r["magnitude_db"]:.3f} | {r["phase_deg"]:.2f} | {int(r["rlc_edges"])} | '
              f'{int(r["rl_edges"])} | {int(r["eps_edges"])} | {int(r["edges"])} | `{r["file"]}` |')

    if args.tex:
        gray = lambda k: '\\rowcolor{gray!15}\n' if k % 2 == 0 else ''
        print('\n% Table II\n\\begin{tabular}{l|rrrr}\n\\toprule\ntarget & best & q25 & median & q75 \\\\\n\\midrule')
        k = 0
        for t, (target, rows) in enumerate(data.items()):
            if t:
                print('\\midrule')
            for name, decimals, values in table_ii(rows):
                print(gray(k) + f'{target} {name} & ' + ' & '.join(f'\\num{{{v:.{decimals}f}}}' for v in values) + ' \\\\')
                k += 1
        print('\\bottomrule\n\\end{tabular}')
        print('\n% Table III\n\\begin{tabular}{l|rrrrrr}\n\\toprule\ntarget  &  $\\overline{\\Delta|\\!\\cdot\\!|}$\\,(\\SI{}{\\decibel}) & '
              '$\\overline{\\Delta\\angle\\!\\cdot}$\\,($^\\circ$) & \\#RLC & \\#RL & \\#$R_\\varepsilon$ & \\#\\,edges \\\\\n\\midrule')
        for k, (name, r) in enumerate(picks):
            if k and k % 2 == 0:
                print('\\midrule')
            print(('\\rowcolor{gray!15}\n' if k % 2 == 0 else '') +
                  f'{name} & \\num{{{r["magnitude_db"]:.3f}}} & \\num{{{r["phase_deg"]:.2f}}} & \\num{{{int(r["rlc_edges"])}}} & '
                  f'\\num{{{int(r["rl_edges"])}}} & \\num{{{int(r["eps_edges"])}}} & \\num{{{int(r["edges"])}}} \\\\')
        print('\\bottomrule\n\\end{tabular}')


if __name__ == '__main__':
    main()
