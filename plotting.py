"""Plot of the network parameters of a fit against the target (Fig. 3 of the paper)."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np


def plot_bode(freq, h, h_hat, path, domain='S'):
    """Magnitude (dB, left axis) and phase (degrees, right axis) of every unique entry (i <= j) of the target h (markers)
    and of the fit h_hat (lines), both (N_f, P, P) numpy arrays at the frequencies freq. The DC point is not drawn (it is
    not on a log axis) and neither are the frequencies where the target does not exist (NaN). The phase of the fit is
    drawn relative to the target's, so that +180 and -180 degrees do not split the line."""
    keep = (freq > 0) & np.isfinite(h).all(axis=(-2, -1))
    f, h, h_hat = freq[keep], h[keep], h_hat[keep]
    P = h.shape[-1]
    fig, axes = plt.subplots(P, P, figsize=(4 * P, 2.2 * P), squeeze=False, sharex=True, sharey=True)
    db = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    for i in range(P):
        for j in range(P):
            ax = axes[i][j]
            if j < i:
                ax.axis('off')
                continue
            target_phase = np.degrees(np.angle(h[:, i, j]))
            fit_phase = target_phase + np.degrees(np.angle(h_hat[:, i, j] * np.conj(h[:, i, j])))
            ax.plot(f, db(h[:, i, j]), '.', color='tab:blue', ms=3)
            ax.plot(f, db(h_hat[:, i, j]), '-', color='tab:blue', lw=1)
            ax.set_xscale('log')
            ax.text(0.03, 0.93, f'{domain}{i + 1}{j + 1}', transform=ax.transAxes, va='top', fontweight='bold')
            twin = ax.twinx()
            twin.plot(f, target_phase, '.', color='tab:red', ms=3)
            twin.plot(f, fit_phase, '-', color='tab:red', lw=1)
            twin.set_yticks(range(-180, 181, 90))
            if i == j == 0:
                ax.set_ylabel('magnitude (dB)', color='tab:blue')
                twin.set_ylabel('phase (deg)', color='tab:red')
            ax.set_xlabel('frequency (Hz)') if i == P - 1 or j == P - 1 else None
            ax.grid(alpha=0.3)
    fig.suptitle(f'{domain}: target (markers) and fit (lines)')
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def write_bode_csv(freq, h, h_hat, path):
    """The data of plot_bode as CSV (for pgfplots): f_Hz, then for every unique entry ij: t<ij>_dB, t<ij>_deg (target),
    f<ij>_dB, f<ij>_deg (fit, its phase unwrapped next to the target's). All frequencies, DC (f = 0) included; NaN where the
    target does not exist or is exactly 0."""
    P = h.shape[-1]
    db = lambda x: 20 * np.log10(np.abs(x) + 1e-30)
    columns, names = [freq], ['f_Hz']
    for i in range(P):
        for j in range(i, P):
            t_phase = np.degrees(np.angle(h[:, i, j]))
            f_phase = t_phase + np.degrees(np.angle(h_hat[:, i, j] * np.conj(h[:, i, j])))
            exact_zero = h[:, i, j] == 0     # no dB or phase (the entry is NaN there for the same reason)
            columns += [np.where(exact_zero, np.nan, c) for c in (db(h[:, i, j]), t_phase, db(h_hat[:, i, j]), f_phase)]
            names += [f't{i + 1}{j + 1}_dB', f't{i + 1}{j + 1}_deg', f'f{i + 1}{j + 1}_dB', f'f{i + 1}{j + 1}_deg']
    np.savetxt(path, np.column_stack(columns), delimiter=',', header=','.join(names), comments='')
