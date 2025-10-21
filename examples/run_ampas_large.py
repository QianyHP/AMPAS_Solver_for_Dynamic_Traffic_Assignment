import argparse
import os
import sys
from typing import List

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ampas.core import AMPAS
from ampas.oracles import FactoredQuadraticOracle


def make_sparse_quadratic(dim: int, factors: int = 16, seed: int = 0) -> FactoredQuadraticOracle:
	rng = np.random.default_rng(seed)
	# Build a tall skinny factor A (m x n), m=factors, sparse-ish via local filters
	m = max(factors, 1)
	A = np.zeros((m, dim))
	for i in range(m):
		base = rng.standard_normal(dim)
		# Smooth/localize to emulate route interactions
		window = 9
		kernel = np.exp(-0.5 * ((np.arange(window) - window//2)/2.0)**2)
		kernel /= kernel.sum()
		conv = np.convolve(base, kernel, mode='same')
		shift = rng.integers(0, dim)
		A[i] = 0.05 * np.roll(conv, shift)
	g = rng.standard_normal(dim) * 0.05
	return FactoredQuadraticOracle(A=A, g=g)


def run(dim: int, ell: float, eta: float, iters: int, avg_mode: str, eta_schedule: str) -> List[float]:
	oracle = make_sparse_quadratic(dim, factors=32, seed=123)
	ampas = AMPAS(dim=dim, eta=eta, lower_bound=ell, burn_in=8, avg_window=32, avg_mode=avg_mode, eta_schedule=eta_schedule)
	p = np.full(dim, 1.0 / dim)
	state = None
	vals: List[float] = []
	for t in range(iters):
		c = oracle.cost(p)
		p, state = ampas.update(p, c, state)
		vals.append(oracle.objective(p))
	return vals


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n', type=int, default=5000)
	parser.add_argument('--ell', type=float, default=1e-5)
	parser.add_argument('--eta', type=float, default=0.5)
	parser.add_argument('--iters', type=int, default=400)
	parser.add_argument('--avg-mode', type=str, default='window', choices=['window', 'polyak'])
	parser.add_argument('--eta-schedule', type=str, default='sqrt', choices=['const', 'sqrt'])
	parser.add_argument('--save', type=str, default='results/ampas_large.png')
	parser.add_argument('--csv', type=str, default='results/ampas_large.csv')
	parser.add_argument('--no-show', action='store_true')
	args = parser.parse_args()

	vals = run(args.n, args.ell, args.eta, args.iters, args.avg_mode, args.eta_schedule)
	print(f"Final objective: {vals[-1]:.6f}")

	# Save CSV
	if args.csv:
		os.makedirs(os.path.dirname(args.csv), exist_ok=True)
		with open(args.csv, 'w', encoding='utf-8') as f:
			f.write('iter,objective\n')
			for i, v in enumerate(vals, 1):
				f.write(f"{i},{v}\n")

	# Plot
	plt.figure(figsize=(8, 5))
	plt.plot(vals, label='AMPAS')
	plt.xlabel('Iteration')
	plt.ylabel('Objective')
	plt.title(f'AMPAS on Sparse Quadratic (n={args.n})')
	plt.legend()
	plt.tight_layout()
	if args.save:
		os.makedirs(os.path.dirname(args.save), exist_ok=True)
		plt.savefig(args.save, dpi=150)
	if not args.no_show:
		plt.show()


if __name__ == '__main__':
	main()


