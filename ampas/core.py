from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .kl import kl_mirror_step
from .simplex import project_simplex_shrink, ActiveSetState
from .mix import analytic_mix
from .averaging import TailAverager


@dataclass
class AMPASState:
	"""Holds the internal state across AMPAS iterations for warm starts."""
	active_set: Optional[ActiveSetState] = None
	averager: Optional[TailAverager] = None


class AMPAS:
	"""Adaptive Mirror–Projection with Active-set Shrinking on bounded simplex.

	Usage:
	  ampas = AMPAS(dim=n, eta=..., lower_bound=..., burn_in=8, avg_window=16)
	  p = p0  # feasible
	  state = None
	  for t in range(T):
		  p, state = ampas.update(p, cost, state)
	"""

	def __init__(
		self,
		dim: int,
		eta: float,
		lower_bound: float = 0.0,
		burn_in: int = 8,
		avg_window: int = 16,
		proj_batch: int = 64,
		proj_sweeps: int = 6,
		avg_mode: str = "window",
		eta_schedule: str = "const",  # "const" or "sqrt"
	):
		self.dim = int(dim)
		self.eta = float(eta)
		self.lower_bound = float(lower_bound)
		self.burn_in = int(burn_in)
		self.avg_window = int(avg_window)
		self.proj_batch = int(proj_batch)
		self.proj_sweeps = int(proj_sweeps)
		self.avg_mode = avg_mode
		self.eta_schedule = eta_schedule
		self._t = 0

		if self.dim <= 0:
			raise ValueError("dim must be positive")
		if self.lower_bound * self.dim > 1.0 + 1e-12:
			raise ValueError("Infeasible lower_bound: n * ell must be <= 1")

	def _ensure_state(self, state: Optional[AMPASState]) -> AMPASState:
		if state is None:
			state = AMPASState()
		if state.active_set is None:
			state.active_set = ActiveSetState()
		if state.averager is None:
			state.averager = TailAverager(self.dim, self.burn_in, self.avg_window, mode=self.avg_mode)
		return state

	def update(self, p_old: np.ndarray, cost: np.ndarray, state: Optional[AMPASState] = None) -> tuple[np.ndarray, AMPASState]:
		"""One AMPAS iteration.

		Returns the new point and updated state. Output is already feasible.
		"""
		p_old = np.asarray(p_old, dtype=float)
		cost = np.asarray(cost, dtype=float)
		if p_old.size != self.dim or cost.size != self.dim:
			raise ValueError("Dimension mismatch in AMPAS.update")
		state = self._ensure_state(state)

		# (1) KL mirror step
		eta = self.eta if self.eta_schedule == "const" else self.eta / max(1.0, np.sqrt(self._t + 1.0))
		p_kl = kl_mirror_step(p_old, cost, eta, self.lower_bound, eps=0.0)

		# (2) Euclidean projection with active-set shrinking
		z = p_old - eta * cost
		p_eu, new_as_state = project_simplex_shrink(
			z,
			self.lower_bound,
			state=state.active_set,
			batch_size=self.proj_batch,
			max_sweeps=self.proj_sweeps,
			adaptive_batch=True,
			freeze_after=3,
			freeze_duration=5,
			fallback="bisect",
		)
		state.active_set = new_as_state

		# (3) Analytic mixing + single backtracking
		p_new = analytic_mix(p_eu, p_kl, cost, p_old, eta, backtrack=True, max_backtracks=2)

		# (4) Tail averaging (optional)
		p_avg = state.averager.update(p_new)
		self._t += 1
		return p_avg, state


