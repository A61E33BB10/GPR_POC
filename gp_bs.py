"""
Gaussian Process Regression for Black-Scholes Call Option Pricing
=================================================================

Can a GP learn to interpolate BS call prices from a grid of training points?
Spoiler: yes, extremely well.

Black-Scholes with r=0, q=0
----------------------------
The BS call formula simplifies beautifully when r=q=0:

    C = S·Φ(d₊) - K·Φ(d₋)

where
    d₊ = [ln(S/K) + ½σ²T] / (σ√T)
    d₋ = d₊ - σ√T

No discounting (e^{-rT} = 1), no dividend adjustment. Just geometry of lognormal diffusion.
Put-call parity becomes: C - P = S - K (trivially, since PV(K) = K when r=0).
"""

import time
import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import joblib


# ── Black-Scholes ──────────────────────────────────────────────────────────

def bs_call(S, K, T, sigma):
    """BS European call price with r=0, q=0."""
    d_plus = (np.log(S / K) + 0.5 * sigma**2 * T) / (sigma * np.sqrt(T))
    d_minus = d_plus - sigma * np.sqrt(T)
    return S * norm.cdf(d_plus) - K * norm.cdf(d_minus)


def bs_put(S, K, T, sigma):
    """BS European put via put-call parity: P = C - S + K."""
    return bs_call(S, K, T, sigma) - S + K


# ── Sanity checks ─────────────────────────────────────────────────────────

def sanity_checks():
    print("=" * 60)
    print("SANITY CHECKS")
    print("=" * 60)

    # ATM call: S=K=100, T=1, σ=0.20 → ~7.9656
    c = bs_call(100, 100, 1.0, 0.20)
    print(f"\nATM call (S=K=100, T=1, σ=0.20):  {c:.4f}  (expect ~7.97)")

    # Deep ITM: call ≈ S - K
    c_itm = bs_call(120, 80, 1.0, 0.20)
    print(f"Deep ITM (S=120, K=80, T=1, σ=0.20): {c_itm:.4f}  (intrinsic = 40)")

    # Deep OTM: call ≈ 0
    c_otm = bs_call(80, 120, 0.5, 0.10)
    print(f"Deep OTM (S=80, K=120, T=0.5, σ=0.10): {c_otm:.6f}  (expect ≈ 0)")

    # Put-call parity: C - P = S - K
    S, K, T, sigma = 105, 95, 0.75, 0.25
    c = bs_call(S, K, T, sigma)
    p = bs_put(S, K, T, sigma)
    parity_error = abs((c - p) - (S - K))
    print(f"\nPut-call parity check (S={S}, K={K}):")
    print(f"  C - P = {c - p:.6f},  S - K = {S - K:.1f},  error = {parity_error:.2e}")
    print()


# ── Data generation ────────────────────────────────────────────────────────

# (S, K, T, σ) ranges — defined once, used everywhere
RANGES = [(85, 115), (85, 115), (0.5, 1.5), (0.10, 0.30)]


def bs_prices(X):
    """BS call prices for an (N, 4) array of [S, K, T, σ]."""
    return bs_call(X[:, 0], X[:, 1], X[:, 2], X[:, 3])


def make_grid(n=5):
    """n^4 evenly spaced grid points (cartesian product of all input ranges)."""
    axes = [np.linspace(lo, hi, n) for lo, hi in RANGES]
    return np.array(np.meshgrid(*axes)).T.reshape(-1, 4)


def make_random(n=200, seed=42):
    """n random off-grid points for interpolation testing."""
    rng = np.random.default_rng(seed)
    return np.column_stack([rng.uniform(lo, hi, n) for lo, hi in RANGES])


# ── GP training ────────────────────────────────────────────────────────────

def train_gp(X_train, y_train):
    """
    Fit a GP regressor with RBF kernel.

    RBF is the natural choice: BS prices are smooth functions of all inputs,
    and RBF encodes exactly that prior — infinite differentiability.
    We use one length scale per input (anisotropic) because the inputs have
    very different scales and sensitivities (moneyness matters more than sigma).
    WhiteKernel absorbs any numerical noise in the training targets (there
    shouldn't be any here, but it stabilises the fit).
    """
    scaler = StandardScaler().fit(X_train)
    X_scaled = scaler.transform(X_train)

    kernel = RBF(length_scale=np.ones(4)) + WhiteKernel(noise_level=1e-5)

    gp = GaussianProcessRegressor(
        #kernel=kernel,
        n_restarts_optimizer=5,
        normalize_y=True,
    )
    gp.fit(X_scaled, y_train)

    return gp, scaler


def print_hyperparams(gp):
    print("=" * 60)
    print("GP HYPERPARAMETERS (after fitting)")
    print("=" * 60)
    print(f"Kernel: {gp.kernel_}")
    params = gp.kernel_.get_params()
    # RBF length scales (one per input dimension)
    for key, val in sorted(params.items()):
        if 'length_scale' in key or 'noise_level' in key:
            print(f"  {key}: {val}")
    print()


# ── Evaluation ─────────────────────────────────────────────────────────────

def evaluate(y_true, y_pred):
    abs_err = np.abs(y_pred - y_true)
    mae = np.mean(abs_err)
    max_ae = np.max(abs_err)
    rmse = np.sqrt(np.mean(abs_err**2))
    ss_res = np.sum((y_true - y_pred)**2)
    ss_tot = np.sum((y_true - np.mean(y_true))**2)
    r2 = 1 - ss_res / ss_tot

    # relative error — exclude deep OTM (price < 0.50) to avoid div-by-zero noise
    mask = y_true >= 0.50
    rel_err = abs_err[mask] / y_true[mask]

    print("=" * 60)
    print("PREDICTION ACCURACY")
    print("=" * 60)
    print(f"  MAE:             {mae:.6f}")
    print(f"  Max AE:          {max_ae:.6f}")
    print(f"  RMSE:            {rmse:.6f}")
    print(f"  R²:              {r2:.8f}")
    print(f"  Mean rel error:  {np.mean(rel_err):.6f}  ({np.mean(rel_err)*100:.4f}%)")
    print(f"  Max rel error:   {np.max(rel_err):.6f}  ({np.max(rel_err)*100:.4f}%)")
    print(f"  (relative errors computed on {mask.sum()}/{len(y_true)} points with price >= 0.50)")
    print()

    return abs_err, rel_err, mask


def print_examples(X_test, y_true, y_pred, n=10):
    print("=" * 60)
    print("EXAMPLE PREDICTIONS")
    print("=" * 60)
    print(f"{'S':>7} {'K':>7} {'T':>6} {'σ':>6} │ {'True':>9} {'Pred':>9} {'Error':>9}")
    print("─" * 60)
    idx = np.linspace(0, len(y_true) - 1, n, dtype=int)
    for i in idx:
        err = y_pred[i] - y_true[i]
        print(f"{X_test[i,0]:7.2f} {X_test[i,1]:7.2f} {X_test[i,2]:6.3f} {X_test[i,3]:6.3f} │ "
              f"{y_true[i]:9.4f} {y_pred[i]:9.4f} {err:+9.6f}")
    print()


# ── Plots ──────────────────────────────────────────────────────────────────

def plot_pred_vs_true(y_true, y_pred):
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(y_true, y_pred, alpha=0.3, s=8, c='steelblue')
    lims = [min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())]
    ax.plot(lims, lims, 'k--', linewidth=1, label='perfect')
    ax.set_xlabel('True BS Price')
    ax.set_ylabel('GP Predicted Price')
    ax.set_title('Predicted vs True Call Prices')
    ax.legend()
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig('plot_pred_vs_true.png', dpi=150)
    print("Saved plot_pred_vs_true.png")


def plot_rel_error_hist(rel_err):
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.hist(rel_err * 100, bins=50, color='steelblue', edgecolor='white', alpha=0.8)
    ax.set_xlabel('Relative Error (%)')
    ax.set_ylabel('Count')
    ax.set_title('Distribution of Relative Errors (prices >= 0.50)')
    fig.tight_layout()
    fig.savefig('plot_rel_error_hist.png', dpi=150)
    print("Saved plot_rel_error_hist.png")


def plot_abs_error_vs_moneyness(X_test, abs_err, mask):
    fig, ax = plt.subplots(figsize=(9, 5))
    moneyness = X_test[mask, 0] / X_test[mask, 1]  # S/K
    T = X_test[mask, 2]
    sc = ax.scatter(moneyness, abs_err[mask], c=T, cmap='viridis', s=10, alpha=0.6)
    fig.colorbar(sc, ax=ax, label='Maturity T (years)')
    ax.set_xlabel('Moneyness (S/K)')
    ax.set_ylabel('Absolute Error')
    ax.set_title('Absolute Error vs Moneyness (coloured by maturity)')
    fig.tight_layout()
    fig.savefig('plot_abs_error_vs_moneyness.png', dpi=150)
    print("Saved plot_abs_error_vs_moneyness.png")


# ── Main ───────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    sanity_checks()

    print("Generating training data...")
    X_train = make_grid(8)
    y_train = bs_prices(X_train)
    print(f"  X_train: {X_train.shape},  y range: [{y_train.min():.4f}, {y_train.max():.4f}]")

    print("Generating test data (200 random points)...")
    X_test = make_random(200)
    y_test = bs_prices(X_test)
    print(f"  X_test:  {X_test.shape},  y range: [{y_test.min():.4f}, {y_test.max():.4f}]\n")

    n = len(X_train)
    print(f"Training GP on {n} points — kernel matrix is {n}×{n} = {n**2:,} entries ({n**2 * 8 / 1e6:.1f} MB float64)...")
    t0 = time.perf_counter()
    gp, scaler = train_gp(X_train, y_train)
    dt = time.perf_counter() - t0
    print(f"  Training took {dt:.1f}s")
    print_hyperparams(gp)

    # save fitted model for reuse without retraining
    joblib.dump({'gp': gp, 'scaler': scaler}, 'gp_bs_model.joblib')
    print("Saved gp_bs_model.joblib")

    y_pred = gp.predict(scaler.transform(X_test))
    print_examples(X_test, y_test, y_pred)

    abs_err, rel_err, mask = evaluate(y_test, y_pred)

    plot_pred_vs_true(y_test, y_pred)
    plot_rel_error_hist(rel_err)
    plot_abs_error_vs_moneyness(X_test, abs_err, mask)
    print("\nDone.")
