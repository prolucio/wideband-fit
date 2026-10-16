# model-lf8: grounded 8-port fit (data/model_e_lf8.npz, DC + 1 Hz – 56.2 MHz, 157 points)

All runs: seeds 100–107, `--deterministic`, 30 000 joint steps, default topology (16 resonant-block nodes, 4 DC-path
nodes, 1056 parameters). "native" = train.py score of the 8-port. "floating" = train.py's floating score (fit and
target both converted with `circuit_model.grounded_to_floating`, the summary.md column "floating"). It is not scored
against data/model_e_lf.npz, which is now a different reduction (plate shorted to GND).

## Experiments

Code versions. **[A]** is the current code: complex-log error skips exactly-0 target entries (15:10), slim guards DC
(14:20), Z(0) is out of the loss (15:48). **[B]** is before the Z(0) change only. **[C]** is before the log-loss fix
and the slim fix. Relative-error-only runs (`--switch-frac 1`) are not affected by the log-loss fix. Z(0) is one of
157 points.

| run | flags | code | native median | native best (seed) | native S/Z/Y median | floating median | floating best (seed) | floating S/Z/Y median | seeds with Zn > 5 dB | passive |
|---|---|---|---|---|---|---|---|---|---|---|
| runs/model-lf8 (baseline) | defaults | A | 23.10 | 21.82 (107) | 36.7/0.77/32.4 | 39.74 | 24.58 (105) | 27.8/71.0/17.5 | 0/8 | 8/8 |
| pre_zdc/model-lf8 | defaults | B | 23.12 | 21.82 (104) | 35.9/0.87/32.7 | 38.29 | 20.95 (107) | 27.3/69.1/11.9 | 0/8 | 8/8 |
| pre_logfix/model-lf8 | defaults | C | 22.19 | 21.74 (104) | 54.1/0.77/10.9 | 57.07 | 16.19 (107) | 43.1/117.5/10.8 | 1/8 | 8/8 |
| rmin4 | `--r-min 1e-4` | B | 21.53 | 21.09 (105) | 47.8/0.65/17.1 | 50.25 | 30.03 (106) | 36.2/106.2/9.2 | 0/8 | 8/8 |
| sw06_rmin4 | `--switch-frac 0.6 --r-min 1e-4` | C | 24.99 | 22.05 (105) | 20.8/2.11/52.1 | 20.65 | 14.16 (107) | 15.6/37.6/9.0 | 3/8 | 8/8 |
| sw09_rmin4 | `--switch-frac 0.9 --r-min 1e-4` | B | 38.68 | 29.85 (101) | 10.5/26.9/69.7 | 10.98 | 7.28 (104) | 6.8/21.0/5.3 | 6/8 | 8/8 |
| relonly | `--switch-frac 1.0` | C (rel only) | 47.42 | 42.73 (104) | 13.1/47.5/80.4 | 13.85 | 6.08 (104) | 8.9/24.4/6.7 | 7/8 | 8/8 |
| relonly_rmin4 | `--switch-frac 1.0 --r-min 1e-4` | C (rel only) | 44.68 | 30.00 (101) | 13.2/16.6/81.2 | 10.57 | 5.50 (104) | 6.7/18.9/5.4 | 4/8 | 8/8 |
| **runs/model-lf8-best** | `--switch-frac 1.0 --r-min 1e-4` | A | 44.95 | 36.65 (101) | 14.4/37.1/91.4 | **10.12** | **4.57 (104)** | 6.0/18.9/5.4 | 5/8 | 8/8 |

On the old 100 MHz band (runs/old_100MHz/) the picture was the same: baseline native about 22 dB with floating 18–58 dB;
`--switch-frac 1` gave floating 5–10 dB. `--pretrain-steps 0`, `--pretrain-fmax 0.5` (DC-only pretrain), `--lr 0.005`
and `--floor-zy 1e-2` / `1e-4` changed nothing that matters.

**What helped, and why.**
- `--switch-frac 1.0` (relative error only): floating median about 39 → 10 dB. The complex-log phase weights every
  entry equally, and 20 of the 36 unique entries per kind are cross-net entries that this topology cannot reproduce
  (next section). In that phase the fit trades the realizable entries for them. It falls into one of two states. In
  the "conducting" state (S about 54 dB, Y about 11 dB), trace and plate are coupled strongly enough to mimic the
  cross-net Y, and the floating conversion is wrong by 57 dB. In the "isolated" state (S 16–30 dB, Y 37–58 dB), the
  floating error is 16–30 dB. The relative error, with its floor at 10⁻³·max for Z/Y and 0.1·max for S, de-weights the
  small cross-net entries, so the fit gets the self blocks right.
- `--r-min 1e-4`: needed on physical grounds. The target's plate conductances between B terminals are up to 1616 S
  (0.62 mΩ), below the default R ≥ 10 mΩ. With the default, every plate DC edge sits at the 10 mΩ bound. Plate Y self
  terms reach only 27 000 against 98 000 (normalized), and the floating DC coupling between the two loops (through the
  plate) comes out wrong. With 1e-4, the floating DC point fits, including Yn13(0).
- The native score is the wrong criterion for picking the best seed here. In model-lf8-best, the native-best seed
  (101) has floating 6.53 dB and the floating-best seed is 104 (4.57 dB). In the baseline, the native-best seed has
  floating 56.6 dB.

**What remains bad with relative error only.** In 5 of 8 seeds, native Zn is 25–69 dB, all of it below about 10 kHz. In
those seeds, private nodes carry series branches near the window corners (L = 1 H, C = 1 mF, f0 ≈ 5 Hz; e.g. seed 100:
common-mode C(1 Hz) = 7.2 µF against the target's 5.70 pF). The relative error saturates at 1 when the prediction is
far too small: |X − X_t|/(|X_t| + φ max) → 1 for |X| ≪ |X_t|, with no gradient. Seed 102 of relonly_rmin4 sits at
rel-Z = 0.98 for f ≤ 100 Hz. Only the complex-log phase could pull Z back, and that phase is spoiled by the
unrealizable entries. The floating cross-loop capacitance is also left out: entries below the floor are ignored, so
floating Zn13, Zn14, … are 20–30 dB high (best/bode_floating_Z.png).

## Why the 8-port fit struggles: diagnosis

Error breakdown, native complex-log error in dB. Classes of unique entries: trace-self is within one trace (6 entries),
plate is among B1..B4 (10), trace-trace is between trace A and trace B (4), trace-plate is between a trace and the
plate (16). Produced by `.claude/jobs/3aaadbc3/tmp/lf8/diag.py`.

| model | S total | S trace-self / plate / trace-trace / trace-plate | Y total | Y trace-self / plate / trace-trace / trace-plate | Z total |
|---|---|---|---|---|---|
| baseline s104 ("conducting") | 54.1 | 38.4 / 4.8 / 81.5 / 84.0 | 10.8 | 4.3 / 8.4 / 16.8 / 13.2 | 0.59 |
| baseline s107 ("isolated") | 23.6 | 12.0 / 0.5 / 46.3 / 36.7 | 50.8 | 0.4 / 8.4 / 54.5 / 95.3 | 1.67 |
| relonly_rmin4 s104 | 14.4 | 0.9 / 0.08 / 51.6 / 19.0 | 112.4 | 0.3 / 2.8 / 180 / 206 | 0.17 |

- **It is not the plate S** (the B terminals joined by about 1600 S). Plate-block S is 0.08–0.5 dB except in the
  conducting state, and plate Y is 2.8 dB with `--r-min 1e-4`.
- **It is not the low-frequency Z.** Native Zn is 0.2–0.9 dB when the fit has not collapsed. Z of the 8-port is the
  common-mode capacitance to GND in every entry: the model gives 5.69 pF against the target's 5.70 pF. That term holds
  no differential information, but it does no harm. Z(0) was set by solver noise (1.5·10⁻¹¹ S). It is now out of the
  loss, and the score never used it.
- **It is the cross-net entries**: 55–84 % of the S error and 54–84 % of the Y error, at every frequency up to about
  1 MHz. This is a **topology limit**:
  - The target's cross-net Y comes from magnetic coupling between galvanically separate nets: Y_AB ≈ −Y_AA·jωM·Y_BB.
    Over 1 Hz–10 MHz, exactly 10 of the 20 cross-net Y entries have Im Y > 0 at every frequency. The entries alternate
    in sign, and they sum to zero per net (A1/A2 against B1..B4).
  - In a network of positive R, L, C without mutual inductance, with no conductive path between the nets (the target
    has none: those entries are exactly 0 at DC), current crosses between nets only through capacitors. To first
    order in ω, the cross-net admittance is then −jω × (a Kron-reduced capacitance Laplacian, whose off-diagonals are
    ≤ 0). So every cross-net Im Y is ≤ 0, and the entries cannot alternate in sign.
  - Checked on all 80 saved 8-port models: of 3708 capacitance-dominated cross-net entries at 1, 10 and 100 Hz, 2 had
    Im > 0 (numerical noise).
  - Magnitude: target |Y_A1B1| = 1.6·10⁻³ S at 1 Hz. As a capacitor that would be 255 µF, which would short trace to
    plate at high frequency. The real coupling is 4·10⁵ times larger than the trace-plate capacitance (13 pF).
  - Inductances from the target, 10 kHz → 10 MHz: trace A Lt = 195 → 146 nH, plate (B1→B2) Lp = 54 → 51 nH,
    trace-plate M = −51 → −48 nH, trace A–trace B M = 54 → 40 nH. Floating loop L = Lt + Lp + 2M = 147 → 101 nH.
    Without M the loop would be Lt + Lp = 249 → 197 nH.
- **Effect on the floating view.** best/bode_floating_Y.png and seeds/model_s104_floating_Y.png show this. The loop
  self-terms (Yn11, Yn12, Yn33, Yn34) and the DC point fit to about 1 dB. The cross-loop terms (Yn13, Yn14, Yn23,
  Yn24) are 12–20 dB too large above about 10 kHz and have the wrong high-frequency phase. Coupling between the loop of
  trace A and the loop of trace B is set by M(trace A, trace B) and M(trace, plate), which this network cannot
  represent. That puts a floor of about 5 dB on floating Y (5.4 dB in every good seed) and about 6 dB on floating S.
- **Loss definition, now fixed.** Before 15:10, the 40 exactly-0 DC entries per kind added about 8 to the
  frequency-averaged log loss of any isolated model (term ≈ 1260 at DC), and nothing to a model whose cross-net DC was
  also exactly 0. After the fix, the baseline is unchanged (native 23.1 dB, floating 39.7 dB median), so this was not
  the cause.

## Terminal-to-GND shunts (model-lf8-best/best = seed 101)

- No DC edge to GND below 1 TΩ. Total conductance to GND at DC is 1.78·10⁻¹¹ S, which is the fixed 10⁻¹² S/node
  leakage of the nodes tied to the terminals (target 1.64·10⁻¹¹ S, solver noise). The floating-best seed 104 has
  9.6·10⁻¹² S.
- Common-mode capacitance to GND is 5.69 pF at 1 Hz and at 1 kHz (target 5.70 pF).
- There are 6 resonant branches to GND, all from private nodes (rlc2: 14.6 µF/1 H, rlc8: 416 µF/1 H, etc.). They sit
  in series with small couplings, so the terminals do not see them. In the Z-broken seeds (e.g. seed 100) such
  branches couple to the terminals: common-mode C(1 Hz) = 7.2 µF and G(1 kHz) = 1.2 mS. Those are the "Zn > 5 dB"
  seeds.
- Passivity: 8/8 seeds pass. The best seed has margins of −1.2·10⁻¹¹ (within the 10⁻⁹ tolerance).

## Proposed changes (not made)

1. **Topology (the main limit): add mutual inductance.** Give the DC-path edges, or a dedicated set of inductive
   branches, a full inductance matrix L = D + W Wᵀ (D ≥ 0 diagonal, W trainable n_edge × r). That keeps the matrix
   positive semidefinite, so the network stays passive and reciprocal. In the nodal stamp, the edge block becomes
   Aᵀ(R + jωL)⁻¹A, solved as one small dense matrix per frequency. This is exactly Q3D's partial-inductance
   description, and it is the only way the grounded 8-port can carry trace-plate and trace-trace M. Without it, the
   floating view of any 8-port fit stays at about 5 dB on Y and the native score at 20 dB or more.
2. **Or fit the floating 4-port directly** when the floating behaviour is what matters. In the floating 4-port, M is
   absorbed into loop inductances that a series R-L branch can represent. On the old band that fit reached about
   0.1 dB.
3. **Loss: a floored complex-log error** |ln((X + φ·max)/(X_t + φ·max))|. It does not saturate when the prediction is
   far too small (the Z-collapse of rel-only seeds), and like the relative error it ignores entries far below the
   largest one (the unrealizable cross-net terms at low frequency).
4. **Seed selection for grounded targets:** choose the best seed by the floating score (or by both), not the native
   score. The native score averages the 20/36 cross-net entries uniformly, so it ranks the floating-worst
   ("conducting") seeds first.
5. The window `--r-min` should lie below the smallest DC resistance of the target. Here that is 0.62 mΩ on the plate,
   so 1e-4 is needed. A rule derived from the target's DC conductance matrix would set this without a flag.

## Files

- Baseline: runs/model-lf8/summary.md, best/bode_{S,Z,Y}.png, best/bode_floating_{S,Z,Y}.png
- Final: runs/model-lf8-best/summary.md, best/ (model.pt, model.cir, bode_*.png, bode_floating_*.png,
  model_floating_*.png from evaluate.py against the exact floating conversion), figures_best/, floating_scores.txt
  (every seed, evaluate.py --floating against the floating conversion of the 8-port target)
- Experiments: runs/tests-lf8/<name>/ with logs <name>.log; earlier code versions in pre_logfix/ and pre_zdc/.
