from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np


class CostOracle(ABC):
	"""Abstract cost oracle mapping p -> cost(p) and objective value if available."""

	@abstractmethod
	def cost(self, p: np.ndarray) -> np.ndarray:
		...

	def objective(self, p: np.ndarray) -> Optional[float]:
		return None


@dataclass
class OperatorState:
	"""Base marker for operator states."""
	pass


class PathUpdateOperator(ABC):
	"""Interface for DTA path update operators on a lower-bounded simplex."""

	def __init__(self, dim: int, lower_bound: float):
		self.dim = int(dim)
		self.lower_bound = float(lower_bound)
		if self.lower_bound * self.dim > 1.0 + 1e-12:
			raise ValueError("Infeasible lower_bound: n * ell must be <= 1")

	@abstractmethod
	def step(self, p_old: np.ndarray, oracle: CostOracle, state: Optional[OperatorState], t: int) -> Tuple[np.ndarray, OperatorState]:
		"""Return (p_new, new_state)."""
		...


