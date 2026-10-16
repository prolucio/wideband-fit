# Fit summary: model_e_hf.npz

- **Target:** `data/model_e_hf.npz`, 4 floating ports, 81 points (1e+05 Hz to 1e+09 Hz), z0 = 50 Ohm.
- **Model:** 16 resonant-block nodes (300 series-RLC branches); 4 DC-path nodes (78 series-RL edges); conductors of the A terminals (initial DC path, from 100000 Hz): {1,2} {3,4}.
- **Settings:** defaults.
- **Seeds:** 100-107 (8), trained batched on NVIDIA L4 in 1267 s (deterministic): pretrain 37 s, graph capture 0 s, joint training 932 s, slimming 38 s, fine-tune 260 s.
- **Slimming:** elements whose removal raises no error component by more than 0.05 dB are removed, then the last 0.33 of the schedule is repeated; elements (resonant branches + DC edges) per seed below, full → slimmed.

## Score

Complex-log error 8.686·|ln(fit/target)| in dB, averaged over the positive frequencies and the unique matrix entries, then over S, Zn, Yn. Lower is better.

| median | 5–95 % range | mean ± std | best |
|---|---|---|---|
| 0.151 dB | 0.098–0.327 dB | 0.184 ± 0.090 dB | 0.082 dB (seed 100) |

Passive (checked on a dense grid from DC up): 8 of 8 seeds.

## Seeds

Low band 1e+05–1e+06 Hz, high band 1e+08–1e+09 Hz (lowest and highest decade of the target; mean over S, Zn, Yn). S, Zn, Yn: whole band.

Out of band: largest curvature |d²ln H/d(ln f)²| of the self-immittances below and above the band (1 Hz to 1e+10 Hz, a factor 3 off the band; 1 = critically damped pair, a resonance of quality Q ~4 Q²), and the number of peaks above 1 there.

| seed | score (dB) | low band | high band | S | Zn | Yn | curv. below | curv. above | peaks | elements | passive |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 100 **best** | 0.082 | 0.015 | 0.275 | 0.072 | 0.088 | 0.085 | 1.54 | 1.9 | 2 | 378 → 197 | yes |
| 107 | 0.128 | 0.003 | 0.467 | 0.123 | 0.135 | 0.128 | 0.998 | 1.99 | 1 | 378 → 200 | yes |
| 105 | 0.140 | 0.008 | 0.480 | 0.141 | 0.162 | 0.116 | 1.01 | 2.39 | 4 | 378 → 200 | yes |
| 102 | 0.147 | 0.002 | 0.530 | 0.150 | 0.122 | 0.168 | 0.495 | 3.12 | 1 | 378 → 203 | yes |
| 103 | 0.156 | 0.002 | 0.561 | 0.147 | 0.158 | 0.163 | 0.465 | 3.75 | 1 | 378 → 222 | yes |
| 101 | 0.175 | 0.006 | 0.596 | 0.183 | 0.167 | 0.175 | 1.71 | 3.36 | 1 | 378 → 213 | yes |
| 106 | 0.304 | 0.003 | 1.072 | 0.321 | 0.303 | 0.287 | 1.32 | 3.42 | 0 | 378 → 218 | yes |
| 104 | 0.339 | 0.004 | 1.228 | 0.387 | 0.305 | 0.327 | 1 | 3.05 | 2 | 378 → 212 | yes |

## Files

- `best/model.cir`: SPICE subcircuit of seed 100 (pins P1A P1B … P4A P4B; port p is the voltage PpA − PpB).
- `best/model.pt`: its checkpoint (`evaluate.py`, `slim.py`, `train.py --init`).
- `best/bode_{S,Z,Y}.png`: target vs fit; `best/loss.png`: training loss; `best/wide.png`: self-immittances and their curvature from the out-of-band low end to the high end, target band shaded.
- `summary.json`: everything above plus the full metrics and loss curve of every seed.
- `seeds/model_s<seed>.{pt,cir}`: checkpoint and netlist of every seed.
