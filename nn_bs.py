"""
Neural Network for Black-Scholes Call Option Pricing
=====================================================

Same task as the GP versions: learn BS call prices from a grid of training points.
But instead of a GP (kernel + Cholesky), we use a plain feedforward neural net.

Why does this work?
-------------------
BS call prices are smooth, continuous, 4D->1D functions. A feedforward net with
ReLU activations is a universal approximator — given enough neurons, it can
approximate any continuous function on a compact domain to arbitrary precision
(Cybenko 1989, Hornik 1991). Our domain is compact (bounded ranges on S,K,T,sigma)
and the function is infinitely differentiable. A small MLP handles this easily.

Architecture choices
--------------------
- 4 inputs (S, K, T, sigma) -> 128 -> 128 -> 128 -> 1 output
- ReLU activations between hidden layers (simplest nonlinearity that works)
- No activation on the output (prices are unbounded positive reals)
- 3 hidden layers: 2 would underfit slightly, 4 adds nothing for a smooth function
- 128 neurons per layer: enough capacity for a 4D smooth surface, small enough to
  train in seconds. Total params: 4*128 + 128*128 + 128*128 + 128*1 + biases ~ 33k

Why normalise?
--------------
Inputs live on very different scales (S~100, sigma~0.2). Without normalisation,
gradients for sigma-weights would be ~500x larger than for S-weights. Normalising
to zero mean / unit variance puts all inputs on equal footing. Same for outputs:
Adam's per-parameter learning rates help, but starting with unit-variance targets
means the initial random predictions are already in the right ballpark.

GP vs NN tradeoff
-----------------
GP: O(N^3) training, gives uncertainty estimates, exact interpolation at data points.
NN: O(epochs * N) training, no uncertainty, but scales to millions of points trivially.
For pure interpolation on 4096 points, the NN trains in seconds vs minutes for the GP.

FP32 everywhere — unlike the GP's CG solver, backprop is numerically stable in float32.
This is the whole point: full GPU throughput on consumer hardware.
"""

import time
import numpy as np
import torch
import torch.nn as nn

from gp_bs import (
    sanity_checks, RANGES, bs_prices, make_grid, make_random,
    evaluate, print_examples,
    plot_pred_vs_true, plot_rel_error_hist, plot_abs_error_vs_moneyness,
)


# ── Device ────────────────────────────────────────────────────────────────
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
DTYPE = torch.float32  # backprop is stable in fp32 — use full GPU throughput


# ── Normalisation ─────────────────────────────────────────────────────────
# Same idea as sklearn StandardScaler, but in torch tensors on the right device.
# We store mean/std so we can apply the same transform at inference time.

def normalise(x):
    mu, sigma = x.mean(dim=0), x.std(dim=0)
    return (x - mu) / sigma, mu, sigma


# ── Model ─────────────────────────────────────────────────────────────────
# nn.Sequential — no class wrapper needed. If your model IS a Sequential,
# wrapping it in a class just adds a forward() that calls self.net(x).
# That's not abstraction, it's bureaucracy.

def make_model():
    return nn.Sequential(
        nn.Linear(4, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 128),
        nn.ReLU(),
        nn.Linear(128, 1),
    )


# ── Training ──────────────────────────────────────────────────────────────

def train(X_np, y_np, epochs=5000, lr=1e-3, batch_size=512):
    """
    Train an MLP on BS call prices.

    Adam optimizer — adaptive per-parameter learning rates handle the fact that
    different weights see very different gradient magnitudes. For this small
    problem, full-batch would work too, but mini-batches add beneficial noise
    that helps escape shallow local minima.

    CosineAnnealingLR decays the learning rate from 1e-3 → 0 following a cosine
    curve. This lets the net make big steps early (explore) and tiny steps late
    (settle into a precise minimum). Eliminates the loss bouncing we saw at
    constant lr.

    Returns the model and normalisation stats needed for inference.
    """
    X = torch.tensor(X_np, dtype=DTYPE, device=DEVICE)
    y = torch.tensor(y_np, dtype=DTYPE, device=DEVICE).unsqueeze(1)

    X_norm, x_mu, x_sig = normalise(X)
    y_norm, y_mu, y_sig = normalise(y)

    model = make_model().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  model: {n_params:,} parameters, device: {DEVICE}")

    n = len(X_norm)
    for epoch in range(epochs):
        # shuffle each epoch — prevents the net from memorising batch order
        perm = torch.randperm(n, device=DEVICE)
        epoch_loss = 0.0

        for i in range(0, n, batch_size):
            idx = perm[i:i+batch_size]
            pred = model(X_norm[idx])
            loss = ((pred - y_norm[idx]) ** 2).mean()  # MSE from first principles

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item() * len(idx)

        scheduler.step()
        epoch_loss /= n
        if epoch % 500 == 0 or epoch == epochs - 1:
            # denormalise the loss for interpretability — this is RMSE in price units
            rmse_price = (epoch_loss ** 0.5) * y_sig.item()
            cur_lr = scheduler.get_last_lr()[0]
            print(f"  epoch {epoch:4d}/{epochs}  MSE(norm)={epoch_loss:.6f}  RMSE(price)={rmse_price:.4f}  lr={cur_lr:.1e}")

    return model, x_mu, x_sig, y_mu, y_sig


def predict(model, X_np, x_mu, x_sig, y_mu, y_sig):
    X = torch.tensor(X_np, dtype=DTYPE, device=DEVICE)
    X_norm = (X - x_mu) / x_sig
    model.eval()
    with torch.no_grad():
        y_norm = model(X_norm)
    # denormalise back to price units
    return (y_norm * y_sig + y_mu).squeeze(1).cpu().numpy()


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
    print(f"Training NN on {n} points ({DEVICE})...")
    t0 = time.perf_counter()
    model, x_mu, x_sig, y_mu, y_sig = train(X_train, y_train)
    dt = time.perf_counter() - t0
    print(f"  Training took {dt:.1f}s\n")

    # save model + normalisation stats
    torch.save({
        'model': model.state_dict(),
        'x_mu': x_mu, 'x_sig': x_sig,
        'y_mu': y_mu, 'y_sig': y_sig,
    }, 'nn_bs_model.pt')
    print("Saved nn_bs_model.pt")

    y_pred = predict(model, X_test, x_mu, x_sig, y_mu, y_sig)
    print_examples(X_test, y_test, y_pred)

    abs_err, rel_err, mask = evaluate(y_test, y_pred)

    plot_pred_vs_true(y_test, y_pred)
    plot_rel_error_hist(rel_err)
    plot_abs_error_vs_moneyness(X_test, abs_err, mask)
    print("\nDone.")
