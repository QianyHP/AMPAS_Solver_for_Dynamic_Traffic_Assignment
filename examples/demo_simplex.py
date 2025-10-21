import numpy as np

from ampas.core import AMPAS


def random_cost(n: int, scale: float = 1.0) -> np.ndarray:
	# Smoothly varying synthetic cost to visualize behavior
	x = np.linspace(0, 2 * np.pi, n, endpoint=False)
	base = np.sin(3 * x) + 0.5 * np.cos(5 * x)
	noise = 0.1 * np.random.randn(n)
	return scale * (base + noise)


def main():
	n = 200
	ell = 1e-4
	eta = 0.5
	T = 100
	ampas = AMPAS(dim=n, eta=eta, lower_bound=ell, burn_in=8, avg_window=16, avg_mode="window", eta_schedule="sqrt")
	# Start from feasible uniform
	p = np.full(n, 1.0 / n)
	state = None
	for t in range(T):
		c = random_cost(n)
		p, state = ampas.update(p, c, state)
		if (t + 1) % 10 == 0:
			print(f"iter={t+1:03d} min={p.min():.3e} max={p.max():.3e} sum={p.sum():.6f}")

	print("Done.")


if __name__ == "__main__":
	main()


