# Method

The method of the paper *Gradient-Based Fitting of Passive-by-Construction Netlists for EM Macromodeling*, with the
details needed to reproduce it. Symbols follow the paper; the file that implements each part is given in brackets. Commands
are in [README.md](README.md).

**Notation.** $P$ ports, $N$ internal nodes, $f$ frequency, $\omega = 2\pi f$, $Z_0 = 50\,\Omega$. $\mathbf S, \mathbf Z, \mathbf Y$ are
$P\times P$ matrices, $h_{ij}$ one entry of one of them, $\hat h_{ij}$ the same entry of the netlist. $\mathbf Z$ and $\mathbf Y$ are
handled normalised, $\mathbf Z/Z_0$ and $Z_0\mathbf Y$, which does not change any loss or error below. $[f_\text{min}, f_\text{max}]$
is the band of the target (positive frequencies). Fixed parameters are in [Table I](#fixed-parameters).

---

## 1. Targets [`sparams.py`, `parse_touchstone.py`]

A target is a file with `freq` (Hz, may contain $f=0$), `params` ($\mathbf S(f)$, complex) and `z0`. $\mathbf Z$ and $\mathbf Y$ follow from
$\mathbf S$ (Eq. 1 of the paper):

$$\mathbf Z = Z_0(\mathbf I+\mathbf S)(\mathbf I-\mathbf S)^{-1}, \qquad \mathbf Y = \tfrac{1}{Z_0}(\mathbf I-\mathbf S)(\mathbf I+\mathbf S)^{-1}.$$

At a frequency where the conversion is ill-defined, $\mathbf Z$ (or $\mathbf Y$) is *missing*: it is left out of the loss and of the
errors there. This happens when
1. $\mathbf I-\mathbf S$ (for $\mathbf Z$) or $\mathbf I+\mathbf S$ (for $\mathbf Y$) is singular: its smallest singular value is below $10^{-12}$ times
   the largest (e.g. $\mathbf Z$ at DC of insulated conductors, where it is infinite);
2. the result is not passive: the smallest eigenvalue of its Hermitian part is below $-10^{-6}$ times the largest in magnitude (the
   data are then rounding noise around a singular matrix);
3. ($\mathbf Z$ at DC only) the smallest eigenvalue of the conductance matrix $\mathrm{Re}\,\mathbf Y(0)$ is below $10^{-9}$ S: a capacitor is open at
   DC, so $\mathbf Z(0)$ exists only through conduction, and a field solver leaves about $10^{-11}$ S between insulated conductors.

The DC point enters the loss and the reported errors (S and Y exist there on the PCB targets).

---

## 2. Trainable netlist [`netlist.py`]

### Topology

$2P$ terminals $P_{1A},\dots,P_{PA},P_{1B},\dots,P_{PB}$ (port $p$ is the pair $(P_{pA}, P_{pB})$), $N$ internal nodes. $P_{1B}$ is the 0 V
reference of the netlist.

- **RLC edges**: a series R-L-C edge between every pair of the $2P+N$ nodes: $(2P+N)(2P+N-1)/2$ edges.
- **RL edges**: a series R-L edge between every pair of the $2P$ terminals: $P(2P-1)$ edges.
- **$R_\varepsilon$**: a fixed resistor from every node to 0 V.

Each edge $n$ has impedance $Z_n = R_n + j\omega L_n + \frac{1}{j\omega C_n}$ (no capacitor for RL edges); the code evaluates the
equivalent $Y_n = j\omega / (z\,\omega_0\,[(1-(\omega/\omega_0)^2) + j(\omega/\omega_0)/Q])$ with $\omega_0 = 1/\sqrt{L_nC_n}$,
$z=\sqrt{L_n/C_n}$, $Q=z/R_n$, which is finite at $\omega=0$. All elements are positive, so the netlist is passive and reciprocal for
any parameter values.

For $P=4$, $N=16$ there are 24 nodes, 276 RLC edges (3 trained parameters each), 28 RL edges (2 each: 884 parameters) and 23 resistors $R_\varepsilon$ (one per node except the reference). The edge counts reported (`scores.csv`, Tables II and III) are those of the exported netlist: edges still present after pruning.

### Element values (Eqs. 3–5)

The trained parameters are $r_n=\log_{10}R_n$, $\ell_n=\log_{10}L_n$ and $c_n=\log_{10}C_n$. The values used are

$$L_n=\min(\max(L_{\min,n}, 10^{\ell_n}), L_\max),\quad C_n=\min(\max(C_\min,10^{c_n}),C_\max),\quad R_n=\min(\max(R_{\min,n},10^{r_n}),R_\max)$$

$$L_{\min,n}=\begin{cases}L_\min & \text{RLC}\\ \max\!\big(L_\min,\ R_n/(2\pi f_\text{DCcap})\big) & \text{RL}\end{cases}\qquad
R_{\min,n}=\begin{cases}\max\!\big(R_\min,\ \sqrt{L_n/C_n}/Q_\text{cap}\big) & \text{RLC}\\ R_\min & \text{RL}\end{cases}$$

so that an RL edge has its cutoff $R_n/(2\pi L_n)\le f_\text{DCcap}$ and an RLC edge $Q_n=\sqrt{L_n/C_n}/R_n\le Q_\text{cap}$. The lower bounds
that depend on other parameters are treated as constants when differentiating. In addition, after every optimiser step the
parameters themselves are clamped to the same ranges (`Netlist.project_`): a parameter that sits beyond a bound would otherwise
have zero gradient and could never come back.

### Solving for $\hat{\mathbf S}$ (nodal analysis, Sec. III-E)

For every frequency: build the nodal admittance matrix of the netlist (edges, $1/R_\varepsilon$ to 0 V, and the termination $1/Z_0$ between
$P_{pA}$ and $P_{pB}$ of every port), drop the reference row and column, and solve for all ports at once with the right-hand
sides $\mathbf j_k$ = current $2/\sqrt{Z_0}$ into $P_{kA}$ and out of $P_{kB}$. With the port voltages $v_{mk}$ and currents
$i_{mk}=\frac{2}{\sqrt{Z_0}}\delta_{mk}-v_{mk}/Z_0$ the waves are $a=(v+Z_0i)/(2\sqrt{Z_0})$, $b=(v-Z_0i)/(2\sqrt{Z_0})$ and
$\hat S_{mk}=b_{mk}/a_{kk}$. Everything is `complex128`, solved with `torch.linalg.solve`. $\hat{\mathbf Z}$ and $\hat{\mathbf Y}$ follow from Eq. 1.
When the RLC edges are disconnected (stage i) the internal nodes are isolated and only the terminals are solved for.

---

## 3. Initialisation [`netlist.py`, `sparams.conductor_groups`]

Each seed draws its own random numbers: `torch.manual_seed(seed)` is called before the netlist is built, and the numbers are
drawn in this order: $f_0$, $z$, $Q$ of the RLC edges, then $R$, $f_L$ of the RL edges.

**RLC edges** (Eq. 6): $\log_{10}f_{0n}\sim\mathcal U(\log_{10}f_\min,\log_{10}f_\max)$, $\log_{10}Q_n\sim\mathcal U(\log_{10}0.05,\log_{10}30)$,
$\log_{10}z_n\sim\mathcal N(2,1)$, then $L_n=z_n/\omega_{0n}$, $C_n=1/(z_n\omega_{0n})$, $R_n=z_n/Q_n$.

**RL edges** (Eq. 7): $\log_{10}R_n\sim\mathcal N(-1,1)$ if the two terminals are galvanically connected, $\mathcal N(13,1)$ otherwise;
$\log_{10}f_{Ln}\sim\mathcal U(4,6)$ and $L_n=R_n/(2\pi f_{Ln})$. The parameters are then clamped as in Section 2.

*Galvanically connected terminals.* Two A terminals are connected if their ports belong to the same *group*; all B terminals are
connected to each other and to the reference (the common plate); an A terminal and a B terminal are not. The groups are read from the
target at its lowest frequency (DC if present, where conduction dominates any capacitive coupling): ports $p\neq q$ are in one group
if $|Y_{pq}|/\sqrt{|Y_{pp}Y_{qq}|}>0.5$ (about 1 for the two ends of one trace, about 0 between insulated traces), groups being merged one pair
after the other. If $\mathbf Y$ does not exist at the lowest frequency, every port is its own group. Starting every RL edge as a good
conductor, or every A terminal as its own conductor, was found to give poor fits.

---

## 4. Training loss [`losses.py`]

At every frequency, for each domain $h\in\{\mathbf S,\mathbf Z,\mathbf Y\}$, the mean over the entries $(i,j)$ of

$$\mathcal L_{\text{rel}\,ij}=\frac{|\hat h_{ij}-h_{ij}|}{|h_{ij}|+\varphi\max_{ij}|h_{ij}|},\qquad
\mathcal L_{\text{ln}\,ij}=\Big|\ln\frac{\hat h_{ij}}{h_{ij}}\Big|=\sqrt{\ln^2\Big|\frac{\hat h_{ij}}{h_{ij}}\Big|+\arg^2\frac{\hat h_{ij}}{h_{ij}}}$$

(Eqs. 8, 9), with the maximum over the entries of the same matrix at the same frequency, $\varphi=0.1$ for $\mathbf S$ and $10^{-3}$ for
$\mathbf Z,\mathbf Y$. The sum of the three domains is averaged over all frequencies, DC included; this is $\mathcal L_\text{rel,tot}$ or $\mathcal L_\text{ln,tot}$.

- The mean runs over all $P^2$ entries, so off-diagonal entries (symmetric by reciprocity) weigh twice as much as diagonal ones.
- $\mathcal L_\text{ln}$ is computed only over entries that are not exactly 0 in the target (the logarithm of 0 does not exist; a solver
  writes exact zeros for the coupling it separates at DC). Those entries stay in $\mathcal L_\text{rel}$.
- Missing $\mathbf Z$ or $\mathbf Y$ (Section 1) are left out at that frequency.
- $\mathcal L_\text{ln}$ has many local minima far from the target, so it is used only after $\mathcal L_\text{rel}$ has brought the fit close.

## 5. Out-of-band penalty [`losses.py`]

At every training step $K=32$ frequencies $f_k$ are drawn, log-uniformly, over $[1\,\text{Hz}, f_\min/3]\cup[3f_\max, 10\,\text{GHz}]$ (one interval is
empty when the band starts below 3 Hz, as for LF; the draw is uniform in $\ln f$ over the total length). At each $f_k$ and its neighbours
$f_ke^{\pm\delta}$ the netlist is evaluated and the curvature of the diagonal entries of $\mathbf Z$ and $\mathbf Y$ is estimated (Eq. 10)

$$\kappa_i(f)=\Big|\frac{d^2\ln\hat h_{ii}}{d(\ln f)^2}\Big|\approx\frac{1}{\delta^2}\Big|\ln\frac{\hat h_{ii}(fe^{\delta})}{\hat h_{ii}(f)}-\ln\frac{\hat h_{ii}(f)}{\hat h_{ii}(fe^{-\delta})}\Big|$$

(a complex logarithm, so the phase counts). A real pole or zero bends $\ln h$ by at most $1/2$, a critically damped pair by $1$, a resonance
of quality $Q$ by about $4Q^2$. The penalty $\mathcal P$ is the mean of $\max(0,\kappa_i(f_k)-1)$ over the $K$ frequencies and the $2P$
diagonal entries (the $\mathbf S$ entries are not used). It is added with weight 0.1 in stage ii.

---

## 6. Training [`train.py`]

Adam ($\beta_2=0.999$, $\epsilon=10^{-8}$, no weight decay) on the parameters $r_n,\ell_n,c_n$; every step uses all target frequencies. Adam
is restarted at the start of each stage.

| stage | what | steps | loss | learning rate |
|---|---|---|---|---|
| i | RL edges only, RLC disconnected, $f\le10$ MHz | 8000 | $\mathcal L_\text{rel,tot}$ | 0.1, cosine to 0 |
| ii | all edges, whole band | 9000, then 21000 | $\mathcal L_\text{rel,tot}+0.1\mathcal P$, then $\mathcal L_\text{ln,tot}+0.1\mathcal P$ | one cycle (below) |
| iii | pruning (Section 7) | | | |
| iv | fine-tuning of the pruned netlist | steps 20000–29999 of stage ii | $\mathcal L_\text{ln,tot}+0.1\mathcal P$ | steps 20000–29999 of the cycle |

Stage ii and iv use `torch.optim.lr_scheduler.OneCycleLR` over the 30000 steps with `max_lr=0.02`, `div_factor=10`, `pct_start=0.1`,
`final_div_factor=800`: the rate rises from 0.002 to 0.02 in 3000 steps (cosine) and falls to $0.002/800=2.5\cdot10^{-6}$ in 27000 steps.
The default cycling of Adam's $\beta_1$ (0.95 → 0.85 → 0.95) is kept. Stage i uses the points with $f\le10.5$ MHz (the 10 MHz point of the
frequency grids, with a margin).

**Parallel seeds.** The $K$ netlists of a run are stacked and evaluated together with `torch.func.vmap`; their losses are summed, so each
seed gets the gradient of its own loss. On a GPU a step is one CUDA-graph replay. The seeds of a run share the $K=32$ out-of-band
frequencies of each step, which are drawn from the random generator after the last seed was initialised; hence the result of a seed
depends on the batch it is run in. To reproduce a seed, rerun the same `--seeds` list, with `--deterministic` for bit-identical results
on the same hardware and software.

## 7. Pruning [`prune.py`]

Applied to every seed after stage ii. The edges (RLC and RL) are sorted by their maximum admittance magnitude over the positive target
frequencies, weakest first, and removed one by one (removed = admittance 0). A removal is kept only if
1. the terminals, the reference node and every node that still has an edge stay connected, and
2. every error component rises by at most 0.005 above its value before pruning. The components are $\overline{|\ln(\hat h/h)|}$ (neper;
   0.005 is 0.043 dB) for each of $\mathbf S,\mathbf Z,\mathbf Y$ over the whole band, its lowest decade, its highest decade, and the DC point (if the target has
   one and $h$ exists there). The DC component prevents pruning from removing the DC path, which no positive-frequency error sees.

This is a test of each component against the unpruned value, not against the previous state and not on the total loss. Finally,
internal nodes with a single neighbour are removed with their edges (no current flows through them). Pruning keeps the netlist passive
and reciprocal.

## 8. Errors and ranking [`metrics.py`, `evaluate.py`, `seed_statistics.py`]

For each domain, over all target frequencies (DC included) and the unique entries ($i\le j$), skipping the entries that are missing
(Section 1) or exactly 0 in the target (their dB and phase do not exist; at DC a solver writes exact zeros for the coupling between
insulated traces):

- magnitude error $\overline{\Delta|\cdot|}$: mean of $|20\log_{10}|\hat h_{ij}|-20\log_{10}|h_{ij}||$ (dB);
- phase error $\overline{\Delta\angle\cdot}$: mean of $|\arg(\hat h_{ij}/h_{ij})|$ (degrees), unweighted.

The reported errors are the means over $\mathbf S$, $\mathbf Z$, $\mathbf Y$. Seeds are ranked by magnitude error only; the median is the lower median
(position $\lfloor(n-1)/2\rfloor+1$ of $n$). The number of edges is #RLC + #RL + #$R_\varepsilon$ (one $R_\varepsilon$ per node except the reference).
Tables II and III of the paper are produced by `seed_statistics.py`; in Table II each column has its own best value and quartiles,
independently of the seed.

## 9. Export and passivity check [`netlist.py`, `metrics.py`]

`seed_<seed>.cir` is a SPICE subcircuit with pins `P1A P1B ... PPA PPB` (port $p$: $P_{pA}$ positive, $P_{pB}$ negative; $P_{1B}$ is the reference
node): for each edge still present, `Rrlc/Lrlc/Crlc` in series or `Rrl/Lrl`, and $R_\varepsilon$ (1 TΩ) from every used node to $P_{1B}$.
`evaluate.py` and `train.py` check passivity on DC, 200 points per decade from $10^{-3}$ Hz to $10^{13}$ Hz and the target frequencies:
$1-\sigma_\max(\mathbf S)^2\ge-10^{-9}$ and the smallest eigenvalue of the Hermitian part of $Z_0\mathbf Y$ at least $-10^{-9}$. This holds by construction;
the check detects numerical problems.

**Transient test** (`transient.py`, Fig. 4). Port 1: ideal voltage source between $P_{1A}$ and $P_{1B}$, square wave 0–10 V, period 1 µs,
50 % duty, 1 ns edges; port 2: $R=10\,\Omega$ in parallel with $C=1$ nF between $P_{2A}$ and $P_{2B}$; ports 3 and 4 open. The script writes a SPICE
deck of this bench and integrates the same circuit (modified nodal equations, BDF2, fixed step 0.1 ns, zero initial state). Checked against
$\hat{\mathbf S}$ for sinusoidal inputs: the steady-state gain and phase agree.

---

## Fixed parameters

| symbol | value | symbol | value | symbol | value |
|---|---|---|---|---|---|
| $N$ | 16 | $L_\min$ | $10^{-16}$ H | $f_\text{DCcap}$ | 1 MHz |
| $P$ | 4 (target) | $L_\max$ | 1 H | $Q_\text{cap}$ | 30 |
| $R_\min$ | 10 mΩ | $C_\min$ | $10^{-20}$ F | $\delta$ | 0.01 |
| $R_\max$ | $10^{15}$ Ω | $C_\max$ | 1 mF | $R_\varepsilon$ | 1 TΩ |
| $\varphi$ | 0.1 ($\mathbf S$), $10^{-3}$ ($\mathbf Z,\mathbf Y$) | $K$ | 32 | $\kappa$ threshold | 1 |

Stage lengths, learning rates and the pruning tolerance are in Section 6 and 7; all are constants at the top of `train.py`, `prune.py`,
`netlist.py` and `losses.py`.
