from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np

from ..core import AMPAS, AMPASState
from .base import PathUpdateOperator, OperatorState, CostOracle


@dataclass
class AMPASOpState(OperatorState):
	state: Optional[AMPASState] = None


class AMPASOperator(PathUpdateOperator):
	def __init__(self, dim: int, lower_bound: float, eta: float = 0.5, avg_mode: str = "window", eta_schedule: str = "sqrt"):
		super().__init__(dim, lower_bound)
		self.algo = AMPAS(dim=dim, eta=eta, lower_bound=lower_bound, avg_mode=avg_mode, eta_schedule=eta_schedule)

	def step(self, p_old: np.ndarray, oracle: CostOracle, state: Optional[OperatorState], t: int) -> Tuple[np.ndarray, OperatorState]:
		st = AMPASOpState() if state is None else state  # type: ignore
		c = oracle.cost(p_old)
		p_new, new_state = self.algo.update(p_old, c, st.state)
		st.state = new_state
		return p_new, st


