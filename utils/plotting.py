import math

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch

from circuit_model import get_phase, get_db


def plot_bode(freqs, S_target, S_pred, path, dpi=150, plot_aspect_ratio=0.5, kind='S'):
    """Grid of |.| / phase panels, one per element ij, target vs fit. Positive frequencies where the target does
    not exist (NaN) are not drawn. At DC such a target (e.g. Z of isolated conductors, infinite) is marked as an
    upward arrow at the top of the panel, and the fit is not drawn there: its value is set only by the model's
    leakage (e.g. Z/z0 ~ 1e10) and its small entries are rounding noise of the S -> Z conversion. A fit value above
    the plotted range is marked as an open upward arrow."""
    ok = torch.isfinite(S_target).flatten(1).all(1) | (freqs == 0)
    freqs, S_target, S_pred = freqs[ok], S_target[ok], S_pred[ok]
    n_ports = S_target.shape[-1]
    # A DC point cannot sit on a log frequency axis: it is drawn as markers in
    # its own slot left of the lowest frequency, past a dotted separator, with
    # a 'DC' tick.
    pos = freqs > 0
    f = freqs[pos]
    dc = torch.nonzero(freqs == 0).flatten()
    dc = dc[0].item() if len(dc) else None
    f_min, f_max = f.min().item(), f.max().item()
    x_dc = f_min / 10**0.75
    x_sep = f_min / 10**0.375

    # same magnitude/phase range on every panel, wide enough to include every
    # element of both target and fit (not just one panel's own values), so
    # panels are visually comparable at a glance
    # ranges from the frequencies where the target exists (the fit of a missing
    # target, e.g. Z at DC through the 1 TOhm node leakage, would swamp them)
    fin = torch.isfinite(S_target).flatten(1).all(1)
    mag_db_all = torch.cat([get_db(S_target[fin]), get_db(S_pred[fin])])
    target_ph_all = get_phase(S_target[fin])
    fit_ph_all = target_ph_all + get_phase(S_pred[fin] * torch.conj(S_target[fin]))
    phase_deg_all = torch.cat([target_ph_all, fit_ph_all]) * 180 / torch.pi
    mag_margin = 0.05 * (mag_db_all.max() - mag_db_all.min())
    mag_ylim = (mag_db_all.min().item() - mag_margin.item(), mag_db_all.max().item() + mag_margin.item())
    phase_margin = 0.05 * (phase_deg_all.max() - phase_deg_all.min())
    phase_ylim = (phase_deg_all.min().item() - phase_margin.item(), phase_deg_all.max().item() + phase_margin.item())

    width = 4.5 * n_ports
    fig, axes = plt.subplots(n_ports, n_ports, figsize=(width, width*plot_aspect_ratio),
                             squeeze=False, sharex=True, sharey=True)

    for i in range(n_ports):
        for j in range(n_ports):
            ax1 = axes[i][j]
            target_ij = S_target[pos, i, j]
            pred_ij = S_pred[pos, i, j]

            ax1.plot(f, get_db(target_ij), color='tab:blue', label=f'target |{kind}|')
            ax1.plot(f, get_db(pred_ij), color='tab:cyan', linestyle='--', label=f'fit |{kind}|')
            ax1.set_xscale('log')
            if dc is not None:
                t_db, p_db = get_db(S_target[dc, i, j]).item(), get_db(S_pred[dc, i, j]).item()
                top = mag_ylim[1] - 0.04 * (mag_ylim[1] - mag_ylim[0])
                if math.isfinite(t_db):
                    ax1.plot(x_dc, t_db, 'o', color='tab:blue', markersize=5)
                    if p_db > mag_ylim[1]:
                        ax1.plot(x_dc, top, '^', mfc='none', color='tab:cyan', markersize=10)
                    else:
                        ax1.plot(x_dc, p_db, 'x', color='tab:cyan', markersize=7)
                else:
                    ax1.plot(x_dc, top, '^', color='tab:blue', markersize=6)
                    ax1.text(x_dc * 10**0.12, top, '∞', color='tab:blue', ha='left', va='center')
                ax1.axvline(x_sep, color='gray', linestyle=':', linewidth=1)
            ax1.set_ylim(*mag_ylim)
            ax1.tick_params(axis='y', labelcolor='tab:blue')
            ax1.text(0.02, 0.95, f'{kind}{i+1}{j+1}', transform=ax1.transAxes,
                     ha='left', va='top', fontsize='medium', fontweight='bold')
            ax1.grid(color='tab:blue', alpha=0.3)

            ax2 = ax1.twinx()
            target_ph = get_phase(target_ij)
            # fit phase drawn relative to the target: +180 and -180 are the
            # same angle, so only a real phase error separates the two curves
            # (the fit may then step slightly outside [-180, 180])
            fit_ph = target_ph + get_phase(pred_ij * torch.conj(target_ij))
            ax2.plot(f, target_ph * 180 / torch.pi, color='tab:red', label='target phase')
            ax2.plot(f, fit_ph * 180 / torch.pi, color='tab:brown', linestyle='--', label='fit phase')
            if dc is not None and torch.isfinite(S_target[dc, i, j]):
                t_dc, p_dc = S_target[dc, i, j], S_pred[dc, i, j]
                # unwrapped next to the lowest plotted frequency, so +-180 do not split
                t_ph = target_ph[0] + get_phase(t_dc * torch.conj(target_ij[0]))
                ax2.plot(x_dc, t_ph * 180 / torch.pi, 'o', color='tab:red', markersize=5)
                ax2.plot(x_dc, (t_ph + get_phase(p_dc * torch.conj(t_dc))) * 180 / torch.pi,
                         'x', color='tab:brown', markersize=7)
            ax2.set_ylim(*phase_ylim)
            ax2.set_yticks(range(-90 * math.ceil(-phase_ylim[0] / 90),
                                 90 * math.floor(phase_ylim[1] / 90) + 1, 90))
            ax2.tick_params(axis='y', labelcolor='tab:red')
            ax2.grid(color='tab:red', alpha=0.3, linestyle='--')

            # keep axis labels/ticks only on the outer edge of the grid so the
            # subplots can be packed tightly without repeating them
            if i == n_ports - 1:
                ax1.set_xlabel('Frequency (Hz)')
            else:
                plt.setp(ax1.get_xticklabels(), visible=False)

            if j == 0:
                ax1.set_ylabel('Magnitude (dB)', color='tab:blue')
            else:
                plt.setp(ax1.get_yticklabels(), visible=False)

            if j == n_ports - 1:
                ax2.set_ylabel('Phase (degrees)', color='tab:red')
            else:
                plt.setp(ax2.get_yticklabels(), visible=False)

            if i == 0 and j == 0:
                lines = [l for l in ax1.get_lines() + ax2.get_lines()
                         if not l.get_label().startswith('_')]   # skip DC markers/separator
                ax1.legend(lines, [l.get_label() for l in lines], loc='lower right', fontsize='small')

    if dc is not None:
        decades = range(math.ceil(math.log10(f_min)), math.floor(math.log10(f_max)) + 1)
        ax = axes[-1][0]   # sharex: ticks and limits apply to every panel
        ax.set_xticks([x_dc] + [10.0**k for k in decades],
                      ['DC'] + [f'$10^{{{k}}}$' for k in decades])
        ax.set_xlim(x_dc / 10**0.25, f_max * 10**0.1)

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.subplots_adjust(hspace=0.05, wspace=0.05)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def plot_loss(losses, path, dpi=150, plot_aspect_ratio=0.5, switch=None):
    """Joint-training loss per step; switch: step where the relative error gives way to the complex-log error."""
    width = 6
    fig, ax = plt.subplots(figsize=(width, width*plot_aspect_ratio))
    ax.plot(losses, color='tab:green')
    if switch is not None and 0 < switch < len(losses):
        ax.axvline(switch, color='gray', linestyle=':', label='relative -> complex-log error')
        ax.legend(fontsize='small')
    ax.set_yscale('log')
    ax.set_xlabel('Step')
    ax.set_ylabel('Loss')
    ax.grid()
    fig.tight_layout()
    fig.savefig(path, dpi=dpi)
    plt.close(fig)
