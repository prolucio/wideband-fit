# Data

The board is a common ground plate with two galvanically isolated copper traces. Ports 1 and 2 are the two ends of
trace A, ports 3 and 4 the two ends of trace B, each between the trace end and the plate below it (port voltage =
trace − plate).

Three targets, from the exports in `../model_e/`:

| file | model | source | ports | band |
|---|---|---|---|---|
| `model_e_hf.npz` | model-hf | HFSS 2024.1 full-wave (`HFSS_4port_V1.s4p`) | 4 floating | 100 kHz – 1 GHz, 81 points (20 per decade) |
| `model_e_lf.npz` | model-lf | Q3D quasi-static, through a circuit SP block (`Q3D_4port_format_1Hz_100MHz.s4p`) | 4, trace end against the ideal plate | DC + 1 Hz – 56.2 MHz, 157 points (20 per decade) |
| `model_e_lf8.npz` | model-lf8 | Q3D Extractor 2024.1 (`Q3D_8port_V1.s8p`) | 8 grounded | DC + 1 Hz – 56.2 MHz, 157 points (20 per decade) |

Fields as written by `parse_touchstone.py`: `freq` (Hz), `params` (frequencies × ports × ports, complex), `kind`
(`'S'`), `z0` (50 Ω), `grounded` (bool).

```bash
python parse_touchstone.py ../model_e/HFSS_4port_V1.s4p -o data/model_e_hf.npz --fmin 1e5 --fmax 1e9
python parse_touchstone.py ../model_e/Q3D_4port_format_1Hz_100MHz.s4p -o data/model_e_lf.npz --fmax 6e7   # then add DC (below)
python parse_touchstone.py ../model_e/Q3D_8port_V1.s8p -o data/model_e_lf8.npz --ports 7,4,8,5,2,1,3,6 --grounded --fmax 6e7
```

## model_e_hf

The HFSS export (DC – 1 GHz, 142 points) cut to 100 kHz – 1 GHz. No correction is applied.

Like the earlier export of the same board, it contains a frequency-independent conductance from each trace to the
plate, 6.33 µS (A) and 9.86 µS (B), and between the traces, −5.3 nS (`tools/fix_dc_leakage.py --groups 1,2 3,4` on the
full file): an artefact of the solver at DC (a dielectric does not conduct; a negative conductance is not passive).
At 100 kHz it is 77 % of the capacitive part of the trace-to-plate admittance (13.0 pF, 20.1 pF), at 1 MHz 8 %, above
10 MHz below 1 %. A model fitted to this target reproduces it in the band, and below the band it extrapolates a
resistive path from trace to plate (about 1.4 kΩ for trace A in the best seed of runs/model-hf). To remove it, correct
the full file (it needs the DC point) and cut afterwards.

## model_e_lf

The Q3D export (1 Hz – 100 MHz, 20 per decade, no DC point) cut to 1 Hz – 60 MHz. The ports are not floating: port p is
the end of a trace against the plate, and the plate is ideal (no resistance or inductance). The file equals
`model_e_lf8` with the four plate terminals shorted to GND (ports A1..A4 against GND), to 8·10⁻¹⁰ in S. So at DC each
trace has its own resistance (Y11 = 11.64 S, trace A; Y33 = 7.70 S, trace B) and the two traces do not couple
(|Y13| = 7·10⁻⁹ S). An earlier, linearly spaced export (DC, then 3.45 MHz steps) had floating ports with the plate
resistance as the common return (Y11 = 11.43 S, Y13 = −0.078 S); runs on it are archived in runs/tests-lf/prefix/.

The DC point is added from `model_e_lf8` by the same reduction (the A block of its DC admittance matrix), which differs
from the 1 Hz point by 1.7·10⁻⁸ in S:

```python
d = dict(np.load('data/model_e_lf.npz')); d8 = np.load('data/model_e_lf8.npz')
Y8 = np.linalg.solve(np.eye(8) + d8['params'][0], np.eye(8) - d8['params'][0])
S_dc = np.linalg.solve(np.eye(4) + Y8[:4, :4], np.eye(4) - Y8[:4, :4])
d['freq'] = np.r_[0.0, d['freq']]; d['params'] = np.concatenate([S_dc[None], d['params']])
np.savez_compressed('data/model_e_lf.npz', **d)
```

Z does not exist at DC (both traces are isolated from the plate). The DC point taken from `model_e_lf8` has a
conductance of about 4·10⁻¹² S from each trace to the reference, solver noise, which would give a finite Z(0) with
Z13 exactly 0; `circuit_model.target_normalized` leaves Z out at DC when a conductor reaches the reference through less
than 1 nS. Likewise for `model_e_lf8` (1.5·10⁻¹¹ S to GND).

The export has 9 significant digits. Below about 100 Hz the traces are almost isolated from the plate, Z is a large
capacitive reactance (12 pF and 21 pF), and its small real part is at the rounding level of the data: at 27 of the
points from 1 to 100 Hz it is negative, Z is not passive, and it is left out (circuit_model.target_normalized). Where it
is kept, the real part still carries about 0.03 rad of phase noise.

## model_e_lf8

The same structure as model_e_lf (export DC – 100 MHz, 162 points, cut to DC – 60 MHz), exported with every terminal as a port against Q3D's reference (GND): 4 terminals
on the plate (Box2), 2 on trace A (Box3), 2 on trace B (Box5). `--ports` orders them as A1..A4 (trace ends), B1..B4
(the plate under each end), so that the floating port p is the pair (Ap, Bp):

| floating port | A terminal (file port) | B terminal (file port) |
|---|---|---|
| 1 | 7 (Box3:Rectangle5) | 2 (Box2:Rectangle6) |
| 2 | 4 (Box3:Rectangle3) | 1 (Box2:Rectangle4) |
| 3 | 8 (Box5:Rectangle7) | 3 (Box2:Rectangle8) |
| 4 | 5 (Box5:Rectangle1) | 6 (Box2:Rectangle2) |

The mapping is the only one of the 2·2·24 candidates for which the floating conversion
(`circuit_model.grounded_to_floating`: port p across (Ap, Bp), GND left floating) reproduces `model_e_lf`: max |ΔS| =
5·10⁻¹⁰ at DC and at 100 MHz (the common frequencies), 2·10⁻³ elsewhere after log-frequency interpolation.

Every conductor floats with respect to GND at DC (the rows of Y(0) of each conductor sum to zero, to 10⁻¹¹ S). The
coupling to GND is a capacitance of 5.70 pF in total, constant from 1 Hz to 10 MHz. So Z of the 8-port exists at
every positive frequency. At DC it is set only by a conductance of about 1.5·10⁻¹¹ S to GND, solver noise, and is
left out (circuit_model.target_normalized, below 1 nS).
