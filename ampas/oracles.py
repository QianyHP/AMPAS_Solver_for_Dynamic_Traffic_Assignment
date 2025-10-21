from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .operators.base import CostOracle


@dataclass
class QuadraticOracle(CostOracle):
	"""Quadratic congestion oracle: f(p) = 0.5 p^T H p + g^T p, cost = grad f.

	H is positive semidefinite; we can model path interactions. For demo, H can be
	diagonal or banded. Supports objective value reporting.
	"""

	H: np.ndarray
	g: np.ndarray

	def cost(self, p: np.ndarray) -> np.ndarray:
		return self.H @ p + self.g

	def objective(self, p: np.ndarray) -> float:
		return 0.5 * float(p @ (self.H @ p)) + float(self.g @ p)


@dataclass
class FactoredQuadraticOracle(CostOracle):
	"""Quadratic oracle with H = A^T A, computed via factor A without forming H.

	Objective: f(p) = 0.5 ||A p||^2 + g^T p, cost = A^T(A p) + g.
	"""

	A: np.ndarray  # shape (m, n)
	g: np.ndarray  # shape (n,)

	def cost(self, p: np.ndarray) -> np.ndarray:
		return self.A.T @ (self.A @ p) + self.g

	def objective(self, p: np.ndarray) -> float:
		Ap = self.A @ p
		return 0.5 * float(Ap @ Ap) + float(self.g @ p)


