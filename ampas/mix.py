from __future__ import annotations

import numpy as np


def _quad_proxy(p: np.ndarray, p_old: np.ndarray, cost: np.ndarray, eta: float) -> float:
	"""Quadratic proxy f(p) = <c,p> + (1/(2eta))||p - p_old||^2."""
	d = p - p_old
	return float(np.dot(cost, p) + 0.5 / eta * np.dot(d, d))


def analytic_mix(
	p_eu: np.ndarray,
	p_kl: np.ndarray,
	cost: np.ndarray,
	p_old: np.ndarray,
	eta: float,
	backtrack: bool = True,
	max_backtracks: int = 2,
) -> np.ndarray:
	"""Analytic 1D minimization between Euclidean projection and KL step.

	Given f(p) = <c,p> + (1/(2eta))||p - p_old||^2, minimize over
	{ p(γ) = p_eu + γ (p_kl - p_eu), γ in [0,1] }.
	"""
	p_eu = np.asarray(p_eu, dtype=float)
	p_kl = np.asarray(p_kl, dtype=float)
	cost = np.asarray(cost, dtype=float)
	p_old = np.asarray(p_old, dtype=float)
	d = p_kl - p_eu
	norm2 = float(np.dot(d, d))
	if norm2 <= 1e-12:
		f_eu = _quad_proxy(p_eu, p_old, cost, eta)
		f_kl = _quad_proxy(p_kl, p_old, cost, eta)
		return p_eu if f_eu <= f_kl else p_kl
	num = -eta * float(np.dot(cost, d)) - float(np.dot(p_eu - p_old, d))
	gamma = max(0.0, min(1.0, num / norm2))
	p_new = p_eu + gamma * d
	if backtrack:
		f_eu = _quad_proxy(p_eu, p_old, cost, eta)
		f_kl = _quad_proxy(p_kl, p_old, cost, eta)
		f_min = min(f_eu, f_kl)
		bt = 0
		while bt < max_backtracks:
			f_new = _quad_proxy(p_new, p_old, cost, eta)
			if f_new <= f_min:
				break
			gamma *= 0.5
			p_new = p_eu + gamma * d
			bt += 1
	return p_new


