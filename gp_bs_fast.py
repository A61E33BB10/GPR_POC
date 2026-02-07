"""
GPyTorch version — same GP, faster engine.

sklearn: Cholesky O(N³) × L-BFGS steps × 6 restarts. Pure CPU.
GPyTorch: Cholesky or CG × L-BFGS steps × restarts. CPU or GPU.

  N <= 5000: Cholesky (exact, fast via PyTorch BLAS)
  N >  5000: CG (approximate but avoids O(N³))

float64 is mandatory — float32 CG diverges on near-noiseless data.
On a compatible GPU, kernel matmuls move to CUDA for further speedup.
Same BS model, same data, same evaluation — only the GP engine changes.
"""
from datetime import datetime
import time
import numpy as np
import torch
import gpytorch
from gpytorch.models import ExactGP
from gpytorch.means import ZeroMean
from gpytorch.kernels import RBFKernel
from gpytorch.likelihoods import GaussianLikelihood
from gpytorch.mlls import ExactMarginalLogLikelihood

# reuse everything except the GP from the sklearn version
from gp_bs import (
    sanity_checks, RANGES, bs_prices, make_grid, make_random,
    evaluate, print_examples,
    plot_pred_vs_true, plot_rel_error_hist, plot_abs_error_vs_moneyness,
)

# torch.cuda.is_available() can return True even when the GPU arch is too old
def _pick_device():
    if torch.cuda.is_available():
        try:
            torch.zeros(1, device='cuda')
            return torch.device('cuda')
        except RuntimeError:
            pass
    return torch.device('cpu')

DEVICE = _pick_device()
DTYPE = torch.float64  # float32 causes CG divergence on near-noiseless data


# ── GP model ──────────────────────────────────────────────────────────────
# GPyTorch requires a class here — this is the one place it earns its keep.
# The class defines the GP prior: constant mean + scaled anisotropic RBF.

class ExactGPModel(ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = ZeroMean()
        # ard_num_dims=4 gives one length scale per input (S, K, T, σ)
        # no ScaleKernel — we normalise y instead (matches sklearn's normalize_y=True)
        self.covar_module = RBFKernel(ard_num_dims=4)

    def forward(self, x):
        return gpytorch.distributions.MultivariateNormal(
            self.mean_module(x), self.covar_module(x)
        )


# ── Normalisation ─────────────────────────────────────────────────────────
# sklearn had StandardScaler; here we do it in torch, same idea

def normalise(X, mean=None, std=None):
    if mean is None:
        mean, std = X.mean(dim=0), X.std(dim=0)
    return (X - mean) / std, mean, std


# ── Training ──────────────────────────────────────────────────────────────

def _fit_once(model, likelihood, X, y, n_iter, lr, cholesky_size):
    """Single L-BFGS run from current parameter values."""
    model.train()
    likelihood.train()
    optimizer = torch.optim.LBFGS(model.parameters(), lr=lr, max_iter=20, line_search_fn='strong_wolfe')
    mll = ExactMarginalLogLikelihood(likelihood, model)

    with gpytorch.settings.max_cholesky_size(cholesky_size):
        for _ in range(n_iter):
            def closure():
                optimizer.zero_grad()
                loss = -mll(model(X), y)
                loss.backward()
                return loss
            loss = optimizer.step(closure)
    return loss.item()


def train_gp(X_np, y_np, n_iter=50, lr=0.1, n_restarts=None):
    """
    Train exact GP via L-BFGS with random restarts (like sklearn).

    Cholesky for N <= 5000, CG for N > 5000.
    Multiple restarts avoid bad local optima in the MLL landscape.
    float64 is essential — float32 CG diverges on near-noiseless data.
    """
    X = torch.tensor(X_np, dtype=DTYPE, device=DEVICE)
    y = torch.tensor(y_np, dtype=DTYPE, device=DEVICE)
    X, x_mean, x_std = normalise(X)
    # normalise y (matches sklearn normalize_y=True) — keeps kernel scale ~1
    y_mean, y_std = y.mean(), y.std()
    y_norm = (y - y_mean) / y_std

    n = len(X)
    use_cholesky = n <= 5000
    cholesky_size = n + 1 if use_cholesky else 0
    solver = 'Cholesky' if use_cholesky else 'CG'
    # fewer restarts for large N — each restart is expensive
    if n_restarts is None:
        n_restarts = 5 if n <= 2000 else 2 if n <= 5000 else 0
    print(f"  solver: {solver}, L-BFGS × {n_restarts + 1} restarts (N={n})")

    best_loss, best_state = float('inf'), None

    for restart in range(n_restarts + 1):
        # noise floor of 1e-5 prevents overfitting to near-noiseless data
        # (without this, the optimizer finds shorter length scales that
        # memorise training points but interpolate poorly)
        likelihood = GaussianLikelihood(
            noise_constraint=gpytorch.constraints.Interval(1e-5, 1e-1)
        ).to(DEVICE, dtype=DTYPE)
        model = ExactGPModel(X, y_norm, likelihood).to(DEVICE, dtype=DTYPE)

        # random init for restarts (first run uses defaults)
        if restart > 0:
            model.covar_module.lengthscale = torch.rand(4, dtype=DTYPE, device=DEVICE) * 4 + 0.5

        loss = _fit_once(model, likelihood, X, y_norm, n_iter, lr, cholesky_size)
        tag = " (best)" if loss < best_loss else ""
        print(f"  restart {restart}: loss={loss:.4f}{tag}")

        if loss < best_loss:
            best_loss = loss
            best_state = (model.state_dict(), likelihood.state_dict())

    # reload best
    likelihood = GaussianLikelihood(
        noise_constraint=gpytorch.constraints.Interval(1e-5, 1e-1)
    ).to(DEVICE, dtype=DTYPE)
    model = ExactGPModel(X, y_norm, likelihood).to(DEVICE, dtype=DTYPE)
    model.load_state_dict(best_state[0])
    likelihood.load_state_dict(best_state[1])

    return model, likelihood, x_mean, x_std, y_mean, y_std


def predict(model, likelihood, X_np, x_mean, x_std, y_mean, y_std):
    X = torch.tensor(X_np, dtype=DTYPE, device=DEVICE)
    X = (X - x_mean) / x_std
    model.eval()
    likelihood.eval()
    n_train = model.train_inputs[0].shape[0]
    with torch.no_grad(), gpytorch.settings.max_cholesky_size(n_train + 1):
        y_norm = likelihood(model(X)).mean.cpu().numpy()
    return y_norm * y_std.item() + y_mean.item()


def print_hyperparams(model, likelihood):
    print("=" * 60)
    print("GP HYPERPARAMETERS (after fitting)")
    print("=" * 60)
    ls = model.covar_module.lengthscale.detach().cpu().numpy().squeeze()
    noise = likelihood.noise.item()
    print(f"  RBF length scales: {ls}")
    print(f"  Noise:             {noise:.2e}")
    print(f"  Device:            {DEVICE}")
    print()


# ── Main ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sanity_checks()

    print("Generating training data...")
    X_train = make_grid(8)
    y_train = bs_prices(X_train)
    print(f"  X_train: {X_train.shape},  y range: [{y_train.min():.4f}, {y_train.max():.4f}]")

    print("Generating test data (2000 random points)...")
    X_test = make_random(2000)
    y_test = bs_prices(X_test)
    print(f"  X_test:  {X_test.shape},  y range: [{y_test.min():.4f}, {y_test.max():.4f}]\n")

    n = len(X_train)
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    print(f"Training GP on {n} points (GPyTorch, {DEVICE})...")
    t0 = time.perf_counter()
    model, likelihood, x_mean, x_std, y_mean, y_std = train_gp(X_train, y_train)
    dt = time.perf_counter() - t0
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"))
    print(f"  Training took {dt:.1f}s")
    print_hyperparams(model, likelihood)

    # save fitted model
    torch.save({
        'model': model.state_dict(),
        'likelihood': likelihood.state_dict(),
        'x_mean': x_mean, 'x_std': x_std,
        'y_mean': y_mean, 'y_std': y_std,
    }, 'gp_bs_model.pt')
    print("Saved gp_bs_model.pt")

    y_pred = predict(model, likelihood, X_test, x_mean, x_std, y_mean, y_std)
    print_examples(X_test, y_test, y_pred)

    abs_err, rel_err, mask = evaluate(y_test, y_pred)

    plot_pred_vs_true(y_test, y_pred)
    plot_rel_error_hist(rel_err)
    plot_abs_error_vs_moneyness(X_test, abs_err, mask)
    print("\nDone.")
