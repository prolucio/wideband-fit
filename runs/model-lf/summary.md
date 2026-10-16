# Fit summary: model_e_lf.npz

- **Target:** `data/model_e_lf.npz`, 4 floating ports, 157 points (1 Hz to 5.623e+07 Hz, plus DC), z0 = 50 Ohm.
- **Model:** 16 resonant-block nodes (300 series-RLC branches); 4 DC-path nodes (78 series-RL edges); conductors of the A terminals (initial DC path, from 0 Hz): {1,2} {3,4}.
- **Settings:** defaults.
- **Seeds:** 100-107 (8), trained batched on NVIDIA L4 in 617 s (deterministic): pretrain 48 s, graph capture 0 s, joint training 418 s, slimming 11 s, fine-tune 141 s.
- **Slimming:** elements whose removal raises no error component by more than 0.05 dB are removed, then the last 0.33 of the schedule is repeated; elements (resonant branches + DC edges) per seed below, full → slimmed.

## Score

Complex-log error 8.686·|ln(fit/target)| in dB, averaged over the positive frequencies and the unique matrix entries, then over S, Zn, Yn. Lower is better.

| median | 5–95 % range | mean ± std | best |
|---|---|---|---|
| 0.035 dB | 0.032–0.044 dB | 0.036 ± 0.005 dB | 0.032 dB (seed 100) |

Passive (checked on a dense grid from DC up): 8 of 8 seeds.

## Seeds

Low band 1–10 Hz, high band 5.62e+06–5.62e+07 Hz (lowest and highest decade of the target; mean over S, Zn, Yn). S, Zn, Yn: whole band.

Out of band: largest curvature |d²ln H/d(ln f)²| of the self-immittances below and above the band (1 Hz to 1e+10 Hz, a factor 3 off the band; 1 = critically damped pair, a resonance of quality Q ~4 Q²), and the number of peaks above 1 there.

| seed | score (dB) | low band | high band | S | Zn | Yn | curv. below | curv. above | peaks | elements | passive |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 **best** | 0.032 | 0.011 | 0.127 | 0.020 | 0.066 | 0.011 | nan | 3.55 | 2 | 378 → 196 | yes |
| 101 | 0.033 | 0.003 | 0.137 | 0.023 | 0.070 | 0.005 | nan | 2.48 | 3 | 378 → 188 | yes |
| 102 | 0.034 | 0.004 | 0.143 | 0.025 | 0.068 | 0.008 | nan | 2.59 | 2 | 378 → 184 | yes |
| 105 | 0.034 | 0.003 | 0.140 | 0.025 | 0.069 | 0.009 | nan | 2.88 | 5 | 378 → 179 | yes |
| 103 | 0.036 | 0.005 | 0.145 | 0.024 | 0.072 | 0.011 | nan | 3.3 | 6 | 378 → 165 | yes |
| 107 | 0.036 | 0.004 | 0.154 | 0.024 | 0.072 | 0.011 | nan | 2.45 | 2 | 378 → 188 | yes |
| 106 | 0.039 | 0.003 | 0.165 | 0.025 | 0.084 | 0.008 | nan | 3.11 | 3 | 378 → 171 | yes |
| 104 | 0.047 | 0.004 | 0.216 | 0.049 | 0.072 | 0.021 | nan | 2.47 | 1 | 378 → 178 | yes |

## Files

- `best/model.cir`: SPICE subcircuit of seed 100 (pins P1A P1B … P4A P4B; port p is the voltage PpA − PpB).
- `best/model.pt`: its checkpoint (`evaluate.py`, `slim.py`, `train.py --init`).
- `best/bode_{S,Z,Y}.png`: target vs fit; `best/loss.png`: training loss; `best/wide.png`: self-immittances and their curvature from the out-of-band low end to the high end, target band shaded.
- `summary.json`: everything above plus the full metrics and loss curve of every seed.
- `seeds/model_s<seed>.{pt,cir}`: checkpoint and netlist of every seed.
