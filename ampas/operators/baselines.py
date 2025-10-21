from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..simplex import project_simplex_shrink
from .base import PathUpdateOperator, OperatorState, CostOracle


@dataclass
class ProjState(OperatorState):
	proj_state: Optional[object] = None


class ProjectionOperator(PathUpdateOperator):
	"""Plain Euclidean projection of z = p - eta c onto lower-bounded simplex."""

	def __init__(self, dim: int, lower_bound: float, eta: float = 0.5):
		super().__init__(dim, lower_bound)
		self.eta = float(eta)

	def step(self, p_old: np.ndarray, oracle: CostOracle, state: Optional[OperatorState], t: int):
		st = ProjState() if state is None else state  # type: ignore
		c = oracle.cost(p_old)
		z = p_old - self.eta * c
		p, new_st = project_simplex_shrink(z, self.lower_bound, state=getattr(st, "proj_state", None))
		st.proj_state = new_st
		return p, st


class MSAOperator(PathUpdateOperator):
	"""Method of Successive Averages: p = (1-α)p + α argmin <c,x>, x in simplex."""

	def __init__(self, dim: int, lower_bound: float):
		super().__init__(dim, lower_bound)

	def step(self, p_old: np.ndarray, oracle: CostOracle, state: Optional[OperatorState], t: int):
		c = oracle.cost(p_old)
		# Greedy descent direction: put all mass on argmin c respecting lower bound
		n = p_old.size
		mass = 1.0 - n * self.lower_bound
		idx = int(np.argmin(c))
		x = np.full(n, self.lower_bound, dtype=float)
		x[idx] += mass
		alpha = 1.0 / max(1, t + 1)
		p = (1.0 - alpha) * p_old + alpha * x
		return p, (state or OperatorState())


class FrankWolfeOperator(PathUpdateOperator):
	"""Deterministic Frank-Wolfe on linearized objective with line search on proxy.

	Uses the Euclidean proximal proxy for the line search along the FW direction.
	"""

	def __init__(self, dim: int, lower_bound: float, eta: float = 0.5):
		super().__init__(dim, lower_bound)
		self.eta = float(eta)

	def step(self, p_old: np.ndarray, oracle: CostOracle, state: Optional[OperatorState], t: int):
		c = oracle.cost(p_old)
		n = p_old.size
		mass = 1.0 - n * self.lower_bound
		idx = int(np.argmin(c))
		s = np.full(n, self.lower_bound, dtype=float)
		s[idx] += mass
		d = s - p_old
		# Quadratic proxy line search gamma in [0,1]
		norm2 = float(np.dot(d, d))
		if norm2 <= 1e-12:
			return p_old.copy(), (state or OperatorState())
		num = -self.eta * float(np.dot(c, d)) - float(np.dot(p_old - p_old, d))  # simplifies: -eta <c,d>
		gamma = max(0.0, min(1.0, num / norm2))
		p = p_old + gamma * d
		return p, (state or OperatorState())


class ExtragradientOperator(PathUpdateOperator):
	"""Simple extragradient: p_half = Proj(p - eta c(p)); p = Proj(p - eta c(p_half))."""

	def __init__(self, dim: int, lower_bound: float, eta: float = 0.5):
		super().__init__(dim, lower_bound)
		self.eta = float(eta)

	def step(self, p_old: np.ndarray, oracle: CostOracle, state: Optional[OperatorState], t: int):
		c = oracle.cost(p_old)
		z = p_old - self.eta * c
		p_half, st1 = project_simplex_shrink(z, self.lower_bound, state=None)
		c2 = oracle.cost(p_half)
		z2 = p_old - self.eta * c2
		p, st2 = project_simplex_shrink(z2, self.lower_bound, state=None)
		return p, (state or OperatorState())


