# Gradient-based fitting of passive-by-construction netlists

Code for fitting a trainable netlist of RL and RLC edges to the S-, Z- and Y-parameters of a multiport network
by gradient descent. The netlist is passive and reciprocal by construction and is exported as a SPICE subcircuit.
The method is described in [METHOD.md](METHOD.md).

## Installation

```bash
git clone https://github.com/prolucio/wideband-fit
cd wideband-fit
pip install -r requirements.txt
```

A GPU is recommended: on a CPU (`CUDA_VISIBLE_DEVICES=`) training is much slower.

## Usage

Train several seeds at once (one netlist per seed; results in `runs/hf/100-131/`):

```bash
python train.py --target data/hf.npz --seeds 100-131 --out runs/hf/100-131 --deterministic
```

`--seeds` takes one seed (`3`), a range (`0-15`) or a list (`0,3,5`). The output directory contains `scores.csv`
(one row per seed), `seed_<seed>.pt` (checkpoint) and `seed_<seed>.cir` (SPICE subcircuit). The result of a seed depends on
the other seeds trained with it (METHOD.md, Section 6): to reproduce it, rerun the same `--seeds` with `--deterministic`.

Errors and plots of saved netlists:

```bash
python src/evaluate.py --target data/hf.npz runs/hf/100-131/seed_100.pt --plots
```

Transient simulation of netlists (square wave on port 1, RC load on port 2; also writes the SPICE deck of the test):

```bash
python src/transient.py runs/lf/100-131/seed_118.cir runs/hf/100-131/seed_102.cir --labels LF HF --out runs/transient
```

Statistics over many seeds (400 seeds in batches of 32):

```bash
for s in $(seq 100 32 499); do
  e=$((s + 31)); [ $e -gt 499 ] && e=499
  python train.py --target data/hf.npz --seeds $s-$e --out runs/hf/$s-$e --deterministic
done
python src/seed_statistics.py --lf runs/lf/*/scores.csv --hf runs/hf/*/scores.csv --tex
```

### Your own data

```bash
python src/parse_touchstone.py model.s4p -o data/model.npz [--fmin 1e5 --fmax 1e9]
python train.py --target data/model.npz --seeds 0-31 --out runs/model
```

`src/parse_touchstone.py` reads S-parameters. The two targets of the paper (`data/hf.npz`, `data/lf.npz`) are described in
[data/README.md](data/README.md).

## Citation

```bibtex
@unpublished{prono2026gradient,
  title     = {Gradient-Based Fitting of Passive-by-Construction Netlists for {EM} Macromodeling},
  author    = {Prono, Luciano and Simone, Silvia and Pareschi, Fabio and Setti, Gianluca},
  note      = {Under review},
  year      = {2026}
}
```

## License

MIT, see [LICENSE](LICENSE).
