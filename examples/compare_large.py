import argparse
import os
import sys
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt

# Ensure project root on path
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

from ampas.operators.ampas_op import AMPASOperator
from ampas.operators.baselines import ProjectionOperator, MSAOperator, FrankWolfeOperator, ExtragradientOperator
from ampas.oracles import FactoredQuadraticOracle


def make_factored_quadratic(dim: int, factors: int = 32, seed: int = 0) -> FactoredQuadraticOracle:
	rng = np.random.default_rng(seed)
	m = max(factors, 1)
	A = np.zeros((m, dim))
	for i in range(m):
		base = rng.standard_normal(dim)
		window = 9
		kernel = np.exp(-0.5 * ((np.arange(window) - window//2)/2.0)**2)
		kernel /= kernel.sum()
		conv = np.convolve(base, kernel, mode='same')
		shift = rng.integers(0, dim)
		A[i] = 0.05 * np.roll(conv, shift)
	g = rng.standard_normal(dim) * 0.05
	return FactoredQuadraticOracle(A=A, g=g)


def run_operator(op, oracle, p0, T: int) -> Tuple[np.ndarray, List[float]]:
	p = p0.copy()
	state = None
	vals: List[float] = []
	for t in range(T):
		p, state = op.step(p, oracle, state, t)
		vals.append(oracle.objective(p))
	return p, vals


def main():
	parser = argparse.ArgumentParser()
	parser.add_argument('--n', type=int, default=5000)
	parser.add_argument('--ell', type=float, default=1e-5)
	parser.add_argument('--eta', type=float, default=0.5)
	parser.add_argument('--iters', type=int, default=800)
	parser.add_argument('--avg-mode', type=str, default='window', choices=['window', 'polyak'])
	parser.add_argument('--eta-schedule', type=str, default='sqrt', choices=['const', 'sqrt'])
	parser.add_argument('--save', type=str, default='results/comparison_large.png')
	parser.add_argument('--csv', type=str, default='results/comparison_large.csv')
	parser.add_argument('--no-show', action='store_true')
	args = parser.parse_args()

	oracle = make_factored_quadratic(args.n, factors=32, seed=42)
	p0 = np.full(args.n, 1.0 / args.n)

	ops = {
		"AMPAS": AMPASOperator(args.n, args.ell, eta=args.eta, avg_mode=args.avg_mode, eta_schedule=args.eta_schedule),
		"Projection": ProjectionOperator(args.n, args.ell, eta=args.eta),
		"MSA": MSAOperator(args.n, args.ell),
		"FW": FrankWolfeOperator(args.n, args.ell, eta=args.eta),
		"EG": ExtragradientOperator(args.n, args.ell, eta=args.eta),
	}

	results: Dict[str, List[float]] = {}
	for name, op in ops.items():
		_, vals = run_operator(op, oracle, p0, args.iters)
		results[name] = vals
		print(f"{name:>10s}: final obj = {vals[-1]:.6f}")

	# Save CSV
	if args.csv:
		os.makedirs(os.path.dirname(args.csv), exist_ok=True)
		with open(args.csv, 'w', encoding='utf-8') as f:
			f.write('iter,' + ','.join(results.keys()) + '\n')
			for i in range(args.iters):
				row = [str(i + 1)] + [f"{results[name][i]}" for name in results.keys()]
				f.write(','.join(row) + '\n')

	# Plot
	plt.figure(figsize=(9, 5))
	for name, vals in results.items():
		plt.plot(vals, label=name)
	plt.xlabel('Iteration')
	plt.ylabel('Objective')
	plt.title(f'Large-scale Comparison (n={args.n}, T={args.iters})')
	plt.legend()
	plt.tight_layout()
	if args.save:
		os.makedirs(os.path.dirname(args.save), exist_ok=True)
		plt.savefig(args.save, dpi=150)
	if not args.no_show:
		plt.show()


if __name__ == '__main__':
	main()


