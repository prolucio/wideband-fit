# Targets

Four-port PCB of the paper (Fig. 1): a ground plate and two traces, ports 1 and 2 at the two ends of one trace, ports 3 and 4
at the two ends of the other, each between the trace end (terminal A) and the plate (terminal B). Reference impedance
50 Ω. Fields of the `.npz` files: `freq` (Hz), `params` (S, frequencies × 4 × 4, complex), `z0`.

| file | target | solver | frequencies |
|---|---|---|---|
| `hf.npz` | HF | HFSS (`HFSS_4port_V1.s4p`) | 81 points, 20 per decade, 100 kHz – 1 GHz |
| `lf.npz` | LF | Q3D (`Q3D_4port_format_1Hz_100MHz.s4p`) | DC and 156 points, 20 per decade, 1 Hz – 56.2 MHz |

```bash
python src/parse_touchstone.py HFSS_4port_V1.s4p -o data/hf.npz --fmin 1e5 --fmax 1e9
python src/parse_touchstone.py Q3D_4port_format_1Hz_100MHz.s4p -o data/lf.npz --fmax 6e7   # then add the DC point
```

The Q3D export has no DC point. It is added from the 8-port Q3D export of the same structure (`Q3D_8port_V1.s8p`, every
terminal a port against Q3D's reference; ports ordered A1..A4, B1..B4 = file ports 7,4,8,5,2,1,3,6), by reducing its DC
admittance matrix to the four ports (trace end against the ideal plate):

```python
import numpy as np
from parse_touchstone import parse_touchstone
d = dict(np.load('data/lf.npz'))
_, S8, _ = parse_touchstone('Q3D_8port_V1.s8p')
order = [6, 3, 7, 4, 1, 0, 2, 5]
Y8 = np.linalg.solve(np.eye(8) + S8[0][np.ix_(order, order)], np.eye(8) - S8[0][np.ix_(order, order)])
S_dc = np.linalg.solve(np.eye(4) + Y8[:4, :4], np.eye(4) - Y8[:4, :4])
np.savez_compressed('data/lf.npz', freq=np.r_[0.0, d['freq']], params=np.concatenate([S_dc[None], d['params']]), z0=d['z0'])
```

Z does not exist at DC (the traces are insulated from the plate) and at the lowest frequencies of LF, where its real part is
at the rounding level of the data; such entries are left out of the fit and of the errors (METHOD.md, Section 2).
