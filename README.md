# Passive RLC macromodels by gradient descent

Fits a network of positive resistors, inductors and capacitors to a measured or simulated N-port scattering matrix
S(f). The circuit is passive and reciprocal by construction, and is exported as a SPICE subcircuit.

## Method

**Target.** An N-port S matrix S_t(f_k) on K frequencies (a DC point is allowed), reference impedance z0. The fit also
uses the normalised impedance and admittance matrices Z_n = (I − S)⁻¹(I + S) and Y_n = Z_n⁻¹, derived from S_t. Weak
couplings that S compresses near |S| ≈ 1 or 0 are well conditioned in one of them. Where Z_n or Y_n does not exist
(I − S or I + S singular to working precision, e.g. Z_n at DC when conductors are isolated from each other), it is left
out of the loss and the metrics at that frequency. So is a Z_n or Y_n that is not passive (its Hermitian part has an
eigenvalue below −10⁻⁶ times the largest): that is the rounding noise of the data around a singular matrix (the Q3D
4-port gives Z_n,11 = −3.2·10⁹ at DC). At DC, Z_n also does not exist when a conductor reaches the reference only
through less than 1 nS: a capacitor is open at DC, so Z(0) exists only through conduction, and a solver's isolated
conductors leave a conductance at its rounding level (Q3D: about 10⁻¹¹ S, which gives a finite Z(0) with the
cross-couplings exactly 0).

**Topology** (`circuit_model.py`).
- *Floating ports.* Port p is a terminal pair (p_A, p_B); the source and z0 are connected between them and the port
  voltage is V(p_A) − V(p_B). Terminal 1_B is the reference (ground). With floating ports, positive elements can
  produce sign-inverting transfer terms.
- *Grounded ports* (a target marked `grounded`, e.g. a Q3D export with every terminal as a port). The same network,
  with GND a node of its own instead of 1_B: 2N ports, one per terminal (1_A … N_A, 1_B … N_B), each against GND,
  the solver's reference. The branch lists and parameters are those of the floating model; the branches between a
  terminal and GND, which in the floating model sit in parallel with those to 1_B, become the coupling to the
  reference. `circuit_model.grounded_to_floating` turns a grounded 2N-port S, of the target or of the fit, into the
  floating N-port (port p across (p_A, p_B), GND left floating).
- *Resonant block.* M private nodes (default 16). One series R–L–C branch on every pair of nodes among the 2N terminals,
  these nodes and ground.
- *DC path.* One series R–L edge on every pair of nodes among the terminals and ground (optionally D further private
  nodes, `--dc-nodes`, default 0). Each edge's corner R/(2πL) is at most 1 MHz where L ≤ 1 H allows it (R ≤ 6.3 MΩ).
- *Leakage.* A fixed conductance of 10⁻¹² S (1 TΩ) from every node to ground keeps the nodal matrix regular at DC. It
  is also written into the netlist.
- *Forward model.* For each frequency, the nodal admittance matrix is assembled and solved in complex float64 for all
  port excitations at once, giving S(f).

For a 4-port target this is 300 resonant branches, 36 DC edges and 972 trained parameters; every edge has its own
element values.

**Parameters and constraints.**
- *Trained parameters:* log₁₀ L, C, R per resonant branch; log₁₀ R, L per DC edge.
- *Element window:* every branch satisfies L ≤ 1 H, C ≤ 1 mF, 10 mΩ ≤ R ≤ 10¹⁵ Ω and Q = √(L/C)/R ≤ 30. DC edges
  also satisfy R/(2πL) ≤ 1 MHz up to the L cap. The resistance cap lies far above the 1 TΩ leakage, so
  an element can switch itself off. A cap of 1 GΩ leaves the switched-off edges conducting 1 nS each, which on a board
  with isolated conductors adds tens of nS between conductors that the target separates only by a few pF.
- *Enforcement:* the window is enforced by projecting the parameters after every optimizer step; the forward pass
  applies the same clamps.

**Initialisation** (per seed).
- *Resonant branches:* drawn as (f₀, z = √(L/C), Q) and converted with L = z/ω₀, C = 1/(zω₀), R = z/Q.
  - f₀ is log-uniform over the band of the target.
  - Q is log-uniform over [0.05, 30] (upper end lowered to Q_max if that is below 30).
  - log₁₀ z ~ N(2, 1), with z in Ω.

  The three draws are independent. On the HFSS and low-frequency 4-port targets this fit as well as, or better than,
  a draw that placed f₀ where the target changes fastest and split Q into a wide and a thin half.
- *DC edges:* each edge starts as metal, log₁₀ R ~ N(−1, 1), or as dielectric, log₁₀ R ~ N(13, 1) (far above the
  leakage: off, but trainable); corner f_L log-uniform over [10 kHz, 1 MHz]; L = R/(2π f_L).
  - *Bodies.* Every node belongs to a body: a conductor (the A terminals of its ports), or the ground plate (the B
    terminals and ground, which all ports are measured against; any private DC nodes also start there). An edge
    starts as *metal* if both of its ends are on the same body, and as *dielectric* if they are on different bodies
    (two conductors, or a conductor and the ground plate).
  - *Conductors* are read from the target's lowest frequency (DC if present), where conduction along a conductor
    dominates any capacitive coupling: ports p and q share a conductor when |Y_pq| / √|Y_pp Y_qq| > 0.5 (1.00 for
    the two ends of one trace, 0.007 between the two traces of the Q3D target at DC, where the plate resistance is
    their common return).
    Leaving that out (every A terminal on its own conductor) fails: an edge that starts as dielectric rarely becomes
    metal, so a trace is often never formed.
  - Nothing holds an edge in its class: the training moves it freely. With every edge starting as metal (all terminals
    shorted), the pretrain had to cut its way to a network, and in two seeds out of three it put a port's B terminals
    on the trace and let the reference side float. That state fits everything except the weak coupling between
    conductors, which it gets with the wrong sign, and gradient descent cannot leave it.
  - With grounded ports, GND is the solver's reference and a body of its own, insulated from every conductor: every
    edge to GND starts as dielectric. The conductors are read from the floating conversion of the target.
- Everything is then projected into the window.

**Loss** (`train.py`). For a model entry X and target entry X_t of H ∈ {S, Z_n, Y_n}, let r = X/X_t:
- relative error: |X − X_t| / d, with d = |X_t| + φ·max_ij |X_t| and floor φ = 0.1 for S and 10⁻³ for Z_n, Y_n;
- complex-log error: |ln r| = √(ln²|r| + arg² r), over the entries that are not exactly 0 in the target (a solver
  writes the coupling of conductors it separates exactly as 0 at DC; it has no logarithm and stays in the relative error).

The loss is the mean over frequencies of Σ_H mean_ij of the error. The relative error stays below 1 when the model is
too small, which gives a smooth landscape far from the fit; its floor down-weights entries far below the largest one
in the same matrix. The complex-log error weighs every entry and decade alike, magnitude and phase, but has many local
minima when the fit is far off. With the earlier all-metal DC init, φ = 10⁻³ for S failed on half the seeds and the
complex-log error from the start (pretrain and joint training) on 7 of 8; neither has been retested with the current
init.

**Out-of-band penalty.** Outside the target band the loss says nothing, and a fit can put resonances there. Every step
draws 32 frequency triplets (f e^−h, f, f e^h), f log-uniform over 1 Hz … f_lo/3 and 3·f_hi … 10 GHz (a factor 3 on
each side of the band is left free for resonances whose tails reach into it), and measures the curvature
κ = |d² ln H / d(ln f)²| of the self-immittances diag Z_n, diag Y_n. A real pole or zero bends ln H by at most 1/2, a
critically damped pair by 1, a resonance of quality Q by about Q². The penalty, 0.1 × mean max(κ − 1, 0), is added to
the loss of the joint training: smooth extrapolation is free, peaks are not.

**Schedule.** Adam, full frequency grid at every step, projection after every step.
1. *DC-path pretrain* (8000 steps): resonant block removed, DC-path parameters only, points f ≤ 10.5 MHz,
   relative error, lr 0.1 cosine-annealed to 0. The relative error has no upper bound when the model is too large,
   as against the insulating initial DC path (no capacitors in the pretrain), so the pretrain loss is large (10²–10³)
   and not a measure of the fit.
2. *Joint training, coarse* (first 30 % of 30 000 steps): all parameters, relative error.
3. *Joint training, fine* (remaining 70 %): all parameters, complex-log error.
4. *Slimming* (`slim.py`): elements are removed, weakest first, while no error component (S, Z_n, Y_n; whole band,
   lowest and highest decade, and the DC point if the target has one) rises by more than 0.05 dB and the network stays
   connected. The score leaves DC out, so without the DC component the slimming removed the DC path for free (on the
   Q3D 4-port: traces open at DC in every seed).
5. *Fine-tune*: the slimmed network repeats the last third of the joint schedule.

Stages 2–3 share one one-cycle learning-rate schedule: peak 0.02, 10 % warm-up from 0.002, cosine decay to 2.5·10⁻⁶.
Several seeds are trained at once (`torch.func.vmap` over the stacked parameters; on a GPU each step is one CUDA-graph
replay).

**Score.** 8.686·|ln r| in dB (complex-log error, 1 rad ≙ 8.686 dB), averaged over the positive frequencies and the
unique matrix entries, then over S, Z_n and Y_n. Lower is better. The DC point, when the target has one, is in the
loss but not in the score; its largest relative error per kind is reported as `dc_rel` in `summary.json`.

## Usage

```bash
pip install -r requirements.txt

# train 16 seeds at once (one GPU) on the included target; summary and best model in runs/model_e_hf/
python train.py --target data/model_e_hf.npz --seeds 0-15 --out runs/model_e_hf

# scores, plots, passivity of a saved model
python evaluate.py --target data/model_e_hf.npz runs/model_e_hf/best/model.pt --plots --passivity

# figures of the best seed as PNG + CSV (Bode, wide band, loss, transient, poles, per-seed table)
python tools/export_figures.py runs/model_e_hf --seed best

# slimming at another tolerance, standalone (train.py already slims at 0.05 dB and fine-tunes)
python slim.py runs/model_e_hf/best/model.pt --target data/model_e_hf.npz --tol 0.1 --out runs/model_e_hf/slim
python train.py --target data/model_e_hf.npz --init runs/model_e_hf/slim.pt --start-step 20000 --slim-tol 0 --out runs/model_e_hf_ft

# transient stability of the netlists (SPICE-style trapezoidal and Gear integration; needs scipy)
python tools/netcheck.py runs/model_e_hf/netcheck.json runs/model_e_hf/best/model.cir

# your own target: Touchstone -> .npz (freq, params, z0)
python parse_touchstone.py model.s4p -o data/model.npz

# grounded target (every terminal a port against the reference): order the ports A1..An, B1..Bn and mark it;
# train.py also scores every seed as the floating n-port, evaluate.py --floating against a floating n-port target
python parse_touchstone.py q3d.s8p -o data/model_e_lf8.npz --ports 7,4,8,5,2,1,3,6 --grounded --fmax 6e7
python train.py --target data/model_e_lf8.npz --seeds 0-7 --out runs/lf8
python evaluate.py --target data/model_e_lf.npz --floating runs/lf8/best/model.pt --plots

# field-solver data with a DC point: remove a spurious DC conductance between isolated conductors (data/README.md)
python tools/fix_dc_leakage.py model.s4p --groups 1,2 3,4 -o data/model.npz
```

`train.py` writes to `--out`:
- `summary.md`: target, settings, score statistics over the seeds and one row per seed;
- `summary.json`: the same, plus the full metrics and loss curve of every seed, speed and timings;
- `best/`, for the best seed: `model.cir` (SPICE subcircuit with pins P1A P1B … PnA PnB, plus GND for grounded
  ports), `model.pt` (checkpoint), Bode plots of S, Z_n, Y_n (`bode_{S,Z,Y}.png`; grounded targets also
  `bode_floating_{S,Z,Y}.png`), the loss curve and the wide-band check (`wide.png`: self-immittances and their
  curvature from 1 Hz to 10 GHz).

`--save-all` also keeps the checkpoint and netlist of every seed (`seeds/`). `--deterministic` makes reruns
bit-identical (about 1.3× slower). For a CPU run, hide the GPUs (`CUDA_VISIBLE_DEVICES=`).

Defaults are those of the paper. The element window (L ≤ 1 H, C ≤ 1 mF) admits resonances down to 5 Hz.
`--pretrain-fmax` (default 1.05·10⁷ Hz) should stay below the target's first resonance
divided by the Q cap.

## Results

Seeds 100–107, default settings (30 000 joint steps), `--deterministic`; scores in dB (lower is better).

| target | median | mean ± std | best | passive |
|---|---|---|---|---|
| `model_e_hf` (HFSS, 100 kHz – 1 GHz) | 0.151 | 0.184 ± 0.090 | 0.082 (seed 100) | 8 of 8 |
| `model_e_lf` (Q3D, 4 ports against the ideal plate, DC – 60 MHz) | 0.035 | 0.036 ± 0.005 | 0.032 (seed 100) | 8 of 8 |
| `model_e_lf8` (Q3D, 8 grounded ports, DC – 60 MHz), `--switch-frac 1 --r-min 1e-4` | 44.95 (as floating 4-port: 10.12) | | 36.65 (floating: 4.57) | 8 of 8 |

`model_e_hf`, best seed: below 0.04 dB up to 100 MHz, 0.4–0.5 dB from 300 MHz to 1 GHz, where the narrow notches are
at the right frequencies but slightly off in depth and phase. The target keeps the solver's DC conductance from trace
to plate (data/README.md), so the fit extrapolates a resistive trace-to-plate path below the band.

`model_e_lf`, best seed: 0.01 dB over the lowest decade, 0.13 dB over the top decade, where the error is concentrated.
The DC point (S, Y; Z does not exist there) is matched to 2·10⁻⁴ (largest relative error). 196 of 378 elements remain
after slimming. The fit has a strong resonance just above the band (about 110 MHz) and
does not extrapolate: against the export's points from 63 to 100 MHz, which it never saw, its error grows from 1 dB
to 4–6 dB.

`model_e_lf8` does not fit: the network has no mutual inductance, and without a conductive path between nets (the
target has none) current crosses between them only through capacitors, so to first order in ω every cross-net Im Y
is ≤ 0. The target's magnetic coupling between galvanically separate nets (trace–plate, trace A–trace B, about
−50 nH and 40–54 nH) alternates in sign and carries 55–84 % of the S and Y error. Hyperparameters move the error
between states but do not remove it (runs/tests-lf8/NOTES.md).

## Data and license

The targets `data/model_e_hf.npz` (HFSS, 4 floating ports), `data/model_e_lf.npz` (Q3D, 4 floating ports) and
`data/model_e_lf8.npz` (Q3D, the same structure as 8 grounded ports), and how they were obtained, are described in
`data/README.md`. The code is released under the MIT license (`LICENSE`).

## Files

| file | content |
|---|---|
| `circuit_model.py` | topology, parameters, projection, forward model (S, floating or grounded ports), grounded → floating conversion, SPICE export, passivity check |
| `train.py` | initialisation sampler, losses, schedule, outputs |
| `slim.py` | greedy element removal under an error tolerance |
| `evaluate.py` | scores and plots of saved models |
| `parse_touchstone.py` | Touchstone → .npz converter (optionally derives Z, Y) |
| `utils/metrics.py`, `utils/plotting.py` | fit metrics, Bode and loss plots |
| `utils/wideband.py` | behaviour far outside the band: curvature, out-of-band peaks, wide-band plot |
| `tools/export_figures.py` | figures of a run's best (or any) seed as PNG plus CSV |
| `tools/make_synth_target.py` | synthetic targets: a measured response plus passive low-frequency resonances |
| `tools/transient_check.py`, `tools/netcheck.py`, `tools/transient_report.py` | transient simulation of the netlists without SPICE |
| `tools/fix_dc_leakage.py` | removes a field solver's spurious DC conductance between isolated conductors |
| `data/model_e_{hf,lf,lf8}.npz` | the targets (data/README.md) |
