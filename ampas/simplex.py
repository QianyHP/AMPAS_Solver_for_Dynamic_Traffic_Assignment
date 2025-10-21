from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


@dataclass
class ActiveSetState:
	"""State for warm-started active-set simplex projection.

	Attributes
	----------
	active_indices : np.ndarray | None
		Indices currently assumed active (positive after projection) on the shifted simplex.
	tau : float | None
		Current threshold such that q_i = max(Z_i - tau, 0) and sum q = s.
	frozen_mask : np.ndarray | None
		Optional mask to temporarily keep coordinates inactive to reduce thrashing.
	flip_counts : np.ndarray | None
		Optional per-coordinate counters for active/inactive flips.
	"""

	active_indices: Optional[np.ndarray] = None
	tau: Optional[float] = None
	frozen_mask: Optional[np.ndarray] = None
	flip_counts: Optional[np.ndarray] = None
	cooldown: Optional[np.ndarray] = None


def _project_simplex_exact(Z: np.ndarray, s: float) -> Tuple[np.ndarray, float, np.ndarray]:
	"""Exact projection onto {q >= 0, sum q = s} via sorting.

	Returns q, tau, active_indices.
	"""
	if s < 0:
		raise ValueError("Simplex radius s must be nonnegative")
	if s == 0:
		return np.zeros_like(Z), float(np.max(Z, initial=0.0)), np.array([], dtype=int)
	# Sort in descending order
	mu = np.sort(Z)[::-1]
	cs = np.cumsum(mu)
	# Find rho = max { j | mu_j - (cs_j - s)/j > 0 }
	j = np.arange(1, Z.size + 1)
	cond = mu - (cs - s) / j
	rho = np.max(np.where(cond > 0)[0]) if np.any(cond > 0) else 0
	tau = (cs[rho] - s) / (rho + 1)
	q = np.maximum(Z - tau, 0.0)
	active = np.flatnonzero(q > 0)
	return q, float(tau), active


def _project_simplex_bisect(Z: np.ndarray, s: float, tol: float = 1e-9, max_iter: int = 64) -> Tuple[np.ndarray, float, np.ndarray]:
	"""Projection onto {q>=0, sum q = s} using bisection on the threshold tau.

	Find tau such that sum(max(Z - tau, 0)) = s.
	"""
	if s <= 0:
		return np.zeros_like(Z), float(np.max(Z, initial=0.0)), np.array([], dtype=int)
	Zmax = float(np.max(Z))
	l = Zmax - s
	u = Zmax
	for _ in range(max_iter):
		m = 0.5 * (l + u)
		g = float(np.sum(np.maximum(Z - m, 0.0)))
		if abs(g - s) <= tol:
			l = m
			break
		if g > s:
			l = m
		else:
			u = m
	tau = l
	q = np.maximum(Z - tau, 0.0)
	active = np.flatnonzero(q > 0)
	return q, float(tau), active


def project_simplex_shrink(
	z: np.ndarray,
	lower_bound: float,
	state: Optional[ActiveSetState] = None,
	batch_size: int = 64,
	max_sweeps: int = 6,
	eps: float = 1e-12,
	adaptive_batch: bool = True,
	freeze_after: int = 3,
	freeze_duration: int = 5,
	fallback: str = "sort",
	bisect_tol: float = 1e-9,
) -> Tuple[np.ndarray, ActiveSetState]:
	"""Project z onto the lower-bounded simplex {p >= ell, 1^T p = 1}.

	This uses an active-set with shrinking and small-batch augment, with warm-start
	from the provided state, and falls back to exact sorting-based projection when
	needed.

	Parameters
	----------
	z : np.ndarray
		Unconstrained point to project (size n).
	lower_bound : float
		Lower bound ell for each coordinate; requires n*ell <= 1.
	state : ActiveSetState | None
		Warm-start state; will be updated and returned.
	batch_size : int
		Number of new candidates to add per sweep from the complement.
	max_sweeps : int
		Maximum outer sweeps of shrink/augment before falling back to exact.
	eps : float
		Numerical tolerance.

	Returns
	-------
	p : np.ndarray
		Projected point on the lower-bounded simplex.
	new_state : ActiveSetState
		Updated warm-start state.
	"""
	z = np.asarray(z, dtype=float)
	n = z.size
	if n == 0:
		return z.copy(), ActiveSetState()
	if lower_bound * n - 1.0 > eps:
		raise ValueError("Infeasible lower bound: n * ell must be <= 1")

	# Shift to standard simplex projection q in R^n, q>=0, sum q = s
	Z = z - lower_bound
	s = max(0.0, 1.0 - n * lower_bound)

	# Initialize state
	if state is None:
		state = ActiveSetState()
	if state.frozen_mask is None or state.frozen_mask.shape != (n,):
		state.frozen_mask = np.zeros(n, dtype=bool)
	if state.flip_counts is None or state.flip_counts.shape != (n,):
		state.flip_counts = np.zeros(n, dtype=int)
	if state.cooldown is None or state.cooldown.shape != (n,):
		state.cooldown = np.zeros(n, dtype=int)

	# Cooldown step: decrement timers and unfreeze if ready
	if np.any(state.frozen_mask):
		state.cooldown[state.frozen_mask] = np.maximum(0, state.cooldown[state.frozen_mask] - 1)
		unfreeze = state.frozen_mask & (state.cooldown == 0)
		if np.any(unfreeze):
			state.frozen_mask[unfreeze] = False

	# Quick exits
	if s == 0.0:
		return np.full(n, lower_bound, dtype=float), state

	# Build initial active set A
	if state.active_indices is None or state.active_indices.size == 0:
		# Heuristic: take top-k positive Z as initial active
		k0 = min(batch_size, n)
		idx = np.argpartition(-Z, k0 - 1)[:k0]
		idx = idx[Z[idx] > 0]
		A = np.unique(idx)
	else:
		A = np.array(state.active_indices, copy=True)

	A = A[(A >= 0) & (A < n)]
	A = np.unique(A)

	# Outer sweeps: shrink and augment until KKT satisfied
	for _ in range(max_sweeps):
		A_prev = A.copy()
		if A.size == 0:
			# Activate a fresh batch with largest Z
			k = min(batch_size, n)
			A = np.argpartition(-Z, k - 1)[:k]
			A = A[Z[A] > 0]
			if A.size == 0:
				# All Z <= 0, solution is to put all mass on the max(Z)
				q, tau, act = _project_simplex_exact(Z, s)
				p = lower_bound + q
				state.active_indices, state.tau = act, tau
				return p, state

		# Solve tau on current active set (assumed all positive)
		tau = (np.sum(Z[A]) - s) / float(A.size)
		qA = Z[A] - tau
		# Shrink: remove non-positives
		keep = qA > eps
		if not np.all(keep):
			A = A[keep]
			if A.size == 0:
				continue
			# Recompute tau after shrink
			tau = (np.sum(Z[A]) - s) / float(A.size)
			qA = Z[A] - tau

		# Flip counting and freezing to reduce thrashing
		if A_prev.size > 0 or A.size > 0:
			in_prev = np.zeros(n, dtype=bool)
			in_prev[A_prev] = True
			in_now = np.zeros(n, dtype=bool)
			in_now[A] = True
			changed = in_prev ^ in_now
			if np.any(changed):
				state.flip_counts[changed] += 1
				to_freeze = changed & (state.flip_counts >= freeze_after)
				if np.any(to_freeze):
					state.frozen_mask[to_freeze] = True
					state.cooldown[to_freeze] = freeze_duration
					# Remove newly frozen from A
					if A.size:
						A = A[~state.frozen_mask[A]]

		# Check KKT on complement: any Z_j > tau + eps should be in active
		maskA = np.zeros(n, dtype=bool)
		maskA[A] = True
		cand = (~maskA) & (~state.frozen_mask)
		viol = np.flatnonzero(cand & (Z > tau + eps))
		if viol.size == 0:
			# KKT satisfied
			q = np.zeros(n, dtype=float)
			q[A] = Z[A] - tau
			p = lower_bound + q
			state.active_indices = A
			state.tau = float(tau)
			return p, state

		# Augment with a small batch of most violating indices (adaptive batch)
		if adaptive_batch:
			b = min(viol.size, max(1, max(batch_size, int(0.1 * viol.size))))
		else:
			b = min(batch_size, viol.size)
		# take top-b by Z value
		top = np.argpartition(-Z[viol], b - 1)[:b]
		A = np.unique(np.concatenate([A, viol[top]]))

	# Fallback to exact projection if not converged
	if fallback == "bisect":
		q, tau, act = _project_simplex_bisect(Z, s, tol=bisect_tol)
	else:
		q, tau, act = _project_simplex_exact(Z, s)
	p = lower_bound + q
	state.active_indices, state.tau = act, tau
	return p, state


