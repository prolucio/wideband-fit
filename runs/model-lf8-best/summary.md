# Fit summary: model_e_lf8.npz

- **Target:** `data/model_e_lf8.npz`, 8 grounded ports (terminals A1..A4, B1..B4, each against GND), 157 points (1 Hz to 5.623e+07 Hz, plus DC), z0 = 50 Ohm.
- **Model:** 16 resonant-block nodes (300 series-RLC branches); 4 DC-path nodes (78 series-RL edges); conductors of the A terminals (initial DC path, from 0 Hz): {1,2} {3,4}.
- **Settings:** defaults except `--r-min 0.0001`, `--switch-frac 1`.
- **Seeds:** 100-107 (8), trained batched on NVIDIA L4 in 1372 s (deterministic): pretrain 117 s, graph capture 0 s, joint training 897 s, slimming 28 s, fine-tune 329 s.
- **Slimming:** elements whose removal raises no error component by more than 0.05 dB are removed, then the last 0.33 of the schedule is repeated; elements (resonant branches + DC edges) per seed below, full → slimmed.

## Score

Complex-log error 8.686·|ln(fit/target)| in dB, averaged over the positive frequencies and the unique matrix entries, then over S, Zn, Yn. Lower is better.

| median | 5–95 % range | mean ± std | best |
|---|---|---|---|
| 44.951 dB | 37.193–50.671 dB | 44.153 ± 5.725 dB | 36.646 dB (seed 101) |

As the floating 4-port (fit and target converted with `circuit_model.grounded_to_floating`): median 10.120 dB, best seed 6.528 dB (column "floating" below).

Passive (checked on a dense grid from DC up): 8 of 8 seeds.

## Seeds

Low band 1–10 Hz, high band 5.62e+06–5.62e+07 Hz (lowest and highest decade of the target; mean over S, Zn, Yn). S, Zn, Yn: whole band.

Out of band: largest curvature |d²ln H/d(ln f)²| of the self-immittances below and above the band (1 Hz to 1e+10 Hz, a factor 3 off the band; 1 = critically damped pair, a resonance of quality Q ~4 Q²), and the number of peaks above 1 there.

| seed | score (dB) | low band | high band | S | Zn | Yn | floating | curv. below | curv. above | peaks | elements | passive |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 101 **best** | 36.646 | 37.095 | 25.501 | 13.066 | 0.243 | 96.630 | 6.528 | nan | 2.83 | 1 | 378 → 145 | yes |
| 105 | 38.207 | 37.442 | 29.566 | 12.184 | 0.444 | 101.994 | 7.852 | nan | 2.48 | 1 | 378 → 141 | yes |
| 104 | 38.542 | 38.465 | 26.546 | 14.414 | 0.158 | 101.053 | 4.571 | nan | 4.57 | 3 | 378 → 111 | yes |
| 106 | 44.552 | 52.530 | 26.362 | 16.388 | 24.741 | 92.526 | 8.561 | nan | 3.26 | 0 | 378 → 201 | yes |
| 103 | 45.351 | 64.987 | 20.466 | 12.424 | 49.600 | 74.028 | 13.186 | nan | 13.1 | 3 | 378 → 197 | yes |
| 100 | 49.186 | 75.885 | 21.368 | 16.553 | 68.541 | 62.465 | 19.812 | nan | 3.28 | 3 | 378 → 200 | yes |
| 107 | 49.363 | 77.395 | 21.439 | 16.742 | 65.013 | 66.335 | 17.589 | nan | 2.18 | 2 | 378 → 158 | yes |
| 102 | 51.376 | 72.413 | 25.763 | 14.393 | 49.534 | 90.200 | 11.679 | nan | 1.89 | 6 | 378 → 224 | yes |

## Files

- `best/model.cir`: SPICE subcircuit of seed 101 (pins P1A P1B … P4A P4B GND; every pin is a port against GND).
- `best/model.pt`: its checkpoint (`evaluate.py`, `slim.py`, `train.py --init`).
- `best/bode_{S,Z,Y}.png`: target vs fit; `best/loss.png`: training loss; `best/wide.png`: self-immittances and their curvature from the out-of-band low end to the high end, target band shaded.
- `best/bode_floating_{S,Z,Y}.png`: the same as the floating 4-port (fit and target converted).
- `summary.json`: everything above plus the full metrics and loss curve of every seed.
- `seeds/model_s<seed>.{pt,cir}`: checkpoint and netlist of every seed.
