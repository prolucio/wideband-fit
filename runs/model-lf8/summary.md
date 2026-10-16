# Fit summary: model_e_lf8.npz

- **Target:** `data/model_e_lf8.npz`, 8 grounded ports (terminals A1..A4, B1..B4, each against GND), 157 points (1 Hz to 5.623e+07 Hz, plus DC), z0 = 50 Ohm.
- **Model:** 16 resonant-block nodes (300 series-RLC branches); 4 DC-path nodes (78 series-RL edges); conductors of the A terminals (initial DC path, from 0 Hz): {1,2} {3,4}.
- **Settings:** defaults.
- **Seeds:** 100-107 (8), trained batched on NVIDIA L4 in 968 s (deterministic): pretrain 117 s, graph capture 0 s, joint training 680 s, slimming 16 s, fine-tune 154 s.
- **Slimming:** elements whose removal raises no error component by more than 0.05 dB are removed, then the last 0.33 of the schedule is repeated; elements (resonant branches + DC edges) per seed below, full → slimmed.

## Score

Complex-log error 8.686·|ln(fit/target)| in dB, averaged over the positive frequencies and the unique matrix entries, then over S, Zn, Yn. Lower is better.

| median | 5–95 % range | mean ± std | best |
|---|---|---|---|
| 23.105 dB | 21.840–24.439 dB | 23.115 ± 1.024 dB | 21.817 dB (seed 107) |

As the floating 4-port (fit and target converted with `circuit_model.grounded_to_floating`): median 39.744 dB, best seed 56.648 dB (column "floating" below).

Passive (checked on a dense grid from DC up): 8 of 8 seeds.

## Seeds

Low band 1–10 Hz, high band 5.62e+06–5.62e+07 Hz (lowest and highest decade of the target; mean over S, Zn, Yn). S, Zn, Yn: whole band.

Out of band: largest curvature |d²ln H/d(ln f)²| of the self-immittances below and above the band (1 Hz to 1e+10 Hz, a factor 3 off the band; 1 = critically damped pair, a resonance of quality Q ~4 Q²), and the number of peaks above 1 there.

| seed | score (dB) | low band | high band | S | Zn | Yn | floating | curv. below | curv. above | peaks | elements | passive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 107 **best** | 21.817 | 32.672 | 9.192 | 53.863 | 0.916 | 10.671 | 56.648 | nan | 2.72 | 0 | 378 → 174 | yes |
| 104 | 21.882 | 33.177 | 10.265 | 54.182 | 0.675 | 10.789 | 57.398 | nan | 1.32 | 1 | 378 → 95 | yes |
| 100 | 22.479 | 28.883 | 8.481 | 29.625 | 0.721 | 37.091 | 27.025 | nan | 3.89 | 3 | 378 → 167 | yes |
| 103 | 23.013 | 31.023 | 9.178 | 41.921 | 0.628 | 26.489 | 46.518 | nan | 2.28 | 6 | 378 → 183 | yes |
| 102 | 23.197 | 30.781 | 10.620 | 40.722 | 1.213 | 27.656 | 46.802 | nan | 1.65 | 1 | 378 → 204 | yes |
| 105 | 23.766 | 28.874 | 10.072 | 23.780 | 0.633 | 46.885 | 24.582 | nan | 3.98 | 3 | 378 → 139 | yes |
| 101 | 24.199 | 29.466 | 11.039 | 26.522 | 0.812 | 45.262 | 26.533 | nan | 2.7 | 4 | 378 → 163 | yes |
| 106 | 24.569 | 29.878 | 10.305 | 32.627 | 0.978 | 40.100 | 32.969 | nan | 2.28 | 1 | 378 → 174 | yes |

## Files

- `best/model.cir`: SPICE subcircuit of seed 107 (pins P1A P1B … P4A P4B GND; every pin is a port against GND).
- `best/model.pt`: its checkpoint (`evaluate.py`, `slim.py`, `train.py --init`).
- `best/bode_{S,Z,Y}.png`: target vs fit; `best/loss.png`: training loss; `best/wide.png`: self-immittances and their curvature from the out-of-band low end to the high end, target band shaded.
- `best/bode_floating_{S,Z,Y}.png`: the same as the floating 4-port (fit and target converted).
- `summary.json`: everything above plus the full metrics and loss curve of every seed.
- `seeds/model_s<seed>.{pt,cir}`: checkpoint and netlist of every seed.
