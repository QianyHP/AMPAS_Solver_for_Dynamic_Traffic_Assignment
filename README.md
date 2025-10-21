# AMPAS: Adaptive Mirror–Projection with Active-set Shrinking

Python reference implementation of AMPAS for lower-bounded simplex updates:

- KL mirror step (multiplicative weights) with explicit lower bounds
- Euclidean projection via active-set shrinking with warm-start and fallback
- Analytic 1-D mixing with single backtracking
- Optional Polyak–Ruppert tail averaging (window or Polyak mode)

## Install

```bash
pip install -r requirements.txt
```

## Usage

```python
import numpy as np
from ampas.core import AMPAS

n = 100
ell = 1e-4
eta = 0.3
ampas = AMPAS(
    dim=n,
    eta=eta,
    lower_bound=ell,
    burn_in=8,
    avg_window=16,
    avg_mode="window",   # or "polyak"
    eta_schedule="const"  # or "sqrt"
)

p = np.full(n, 1.0 / n)
state = None
for t in range(50):
    c = np.random.randn(n)
    p, state = ampas.update(p, c, state)
```

## Notes

- The projection subroutine maintains an active set and threshold `tau`,
  only revisiting a small subset each step; it falls back to an exact
  sorting-based projection if a few shrinking sweeps fail.
- Tail averaging smooths long-tail variance; choose `avg_mode` in {`window`, `polyak`}.
- Projection uses active-set shrinking with freeze/wake and adaptive batch; falls back
  to bisection-based exact projection for robustness.

## Large-scale AMPAS experiment

```bash
python examples/run_ampas_large.py --n 5000 --ell 1e-5 --eta 0.5 --iters 400 --avg-mode window --eta-schedule sqrt --save results/ampas_large.png --csv results/ampas_large.csv --no-show
```

This runs AMPAS on a sparse quadratic oracle (synthetic but scalable), saving a plot and a CSV of the objective trajectory.


