from __future__ import annotations

import numpy as np


def kl_mirror_step(p_old: np.ndarray, cost: np.ndarray, eta: float, lower_bound: float, eps: float = 0.0) -> np.ndarray:
	"""KL-prox mirror step on the lower-bounded simplex.

	Computes
	  \tilde q_i = (p_old_i - ell)_+ * exp(-eta (c_i - min c))
	  q = \tilde q / sum_j \tilde q_j
	  p_kl = ell * 1 + (1 - n ell) q

	Parameters
	----------
	p_old : np.ndarray
		Previous iterate in the lower-bounded simplex.
	cost : np.ndarray
		Cost vector for the linear term.
	eta : float
		Stepsize.
	lower_bound : float
		Lower bound ell.

	Returns
	-------
	pk : np.ndarray
		Mirror step result in the same feasible set.
	"""
	p_old = np.asarray(p_old, dtype=float)
	cost = np.asarray(cost, dtype=float)
	assert p_old.shape == cost.shape
	n = p_old.size
	# Stabilize with a min-shift for exp
	c_shift = cost - np.min(cost)
	base = np.maximum(p_old - lower_bound, 0.0)
	if eps > 0.0:
		# epsilon injection to avoid starving coordinates near the lower bound
		base = base + eps
	weights = np.exp(-eta * c_shift) * base
	sum_w = float(np.sum(weights))
	if sum_w <= 0.0:
		# Degenerate: fall back to uniform over remaining mass
		q = np.full(n, 1.0 / n, dtype=float)
	else:
		q = weights / sum_w
	pk = lower_bound + (1.0 - n * lower_bound) * q
	return pk


