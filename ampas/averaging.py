from __future__ import annotations

from collections import deque
from typing import Deque, Optional

import numpy as np


class TailAverager:
	"""Polyak–Ruppert tail averaging over a sliding window after burn-in.

	This class maintains a fixed-size window of the most recent vectors and
	returns their arithmetic mean once burn-in has passed. Before burn-in, the
	input vector is returned unchanged.
	"""

	def __init__(self, dimension: int, burn_in: int = 8, window_size: int = 16, mode: str = "window"):
		if burn_in < 0 or window_size <= 0:
			raise ValueError("Invalid burn_in or window_size")
		self.dimension = int(dimension)
		self.burn_in = int(burn_in)
		self.window_size = int(window_size)
		self.mode = mode  # "window" or "polyak"
		self._buffer: Deque[np.ndarray] = deque(maxlen=self.window_size)
		self._t = 0
		self._sum: Optional[np.ndarray] = None

	def reset(self) -> None:
		self._buffer.clear()
		self._t = 0

	def update(self, vector: np.ndarray) -> np.ndarray:
		"""Push a new vector and return the averaged output.

		If iteration count <= burn_in, returns the input vector.
		Otherwise returns the arithmetic mean over the current window.
		"""
		v = np.asarray(vector, dtype=float)
		if v.size != self.dimension:
			raise ValueError("Vector dimension mismatch in TailAverager.update")
		self._t += 1
		if self._t <= self.burn_in:
			return v
		if self.mode == "polyak":
			if self._sum is None:
				self._sum = np.zeros(self.dimension, dtype=float)
			self._sum += v
			k = self._t - self.burn_in
			return self._sum / float(k)
		else:
			self._buffer.append(v)
			stack = np.stack(list(self._buffer), axis=0)
			return np.mean(stack, axis=0)


