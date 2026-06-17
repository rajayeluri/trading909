import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

np.random.seed(42)

# =========================================================
# BLACK-SCHOLES MODEL
# =========================================================
def black_scholes(S, K, T, r, sigma, option="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option == "call":
        return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        return K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)


# =========================================================
# BINOMIAL MODEL
# =========================================================
def binomial(S, K, T, r, sigma, steps=100):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    stock = np.zeros((steps + 1, steps + 1))
    stock[0, 0] = S

    for i in range(1, steps + 1):
        stock[i, 0] = stock[i - 1, 0] * u
        for j in range(1, i + 1):
            stock[i, j] = stock[i - 1, j - 1] * d

    option = np.zeros_like(stock)

    for j in range(steps + 1):
        option[steps, j] = max(stock[steps, j] - K, 0)

    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option[i, j] = np.exp(-r * dt) * (
                p * option[i + 1, j] + (1 - p) * option[i + 1, j + 1]
            )

    return option[0, 0]


# =========================================================
# HESTON MODEL (FIXED FULL PATH VERSION)
# =========================================================
def heston_mc(S0, K, T, r, v0, kappa, theta, xi, rho, steps=100, paths=10000):

    dt = T / steps

    S = np.full(paths, S0)
    v = np.full(paths, v0)

    S_path = np.zeros((paths, steps + 1))
    v_path = np.zeros((paths, steps + 1))

    S_path[:, 0] = S
    v_path[:, 0] = v

    for t in range(1, steps + 1):

        z1 = np.random.normal(size=paths)
        z2 = np.random.normal(size=paths)

        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2

        v = np.abs(
            v + kappa * (theta - v) * dt
            + xi * np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * w2
        )

        S = S * np.exp(
            (r - 0.5 * np.maximum(v, 0)) * dt
            + np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * w1
        )

        S_path[:, t] = S
        v_path[:, t] = v

    payoff = np.maximum(S - K, 0)
    price = np.exp(-r * T) * np.mean(payoff)

    return price, S_path, v_path


# =========================================================
# UNIFIED API
# =========================================================
def price_option(model, **params):

    if model == "bs":
        return black_scholes(**params)

    elif model == "binomial":
        return binomial(**params)

    elif model == "heston":
        return heston_mc(**params)

    else:
        raise ValueError("Unknown model")


# =========================================================
# INPUT PARAMETERS
# =========================================================
S = 100
K = 100
T = 1
r = 0.05
sigma = 0.2

bs_params = dict(S=S, K=K, T=T, r=r, sigma=sigma, option="call")
bin_params = dict(S=S, K=K, T=T, r=r, sigma=sigma, steps=100)

heston_params = dict(
    S0=S, K=K, T=T, r=r,
    v0=0.04,
    kappa=2.0,
    theta=0.04,
    xi=0.3,
    rho=-0.7,
    steps=100,
    paths=5000
)


# =========================================================
# PRICING COMPARISON
# =========================================================
bs_price = price_option("bs", **bs_params)
bin_price = price_option("binomial", **bin_params)
heston_price, S_path, v_path = price_option("heston", **heston_params)

print("\n==============================")
print(" UNIFIED DERIVATIVES PLATFORM ")
print("==============================")

print(f"Black-Scholes: {bs_price:.4f}")
print(f"Binomial:      {bin_price:.4f}")
print(f"Heston MC:     {heston_price:.4f}")


# =========================================================
# 1. MODEL COMPARISON BAR CHART
# =========================================================
models = ["BS", "Binomial", "Heston"]
prices = [bs_price, bin_price, heston_price]

plt.figure(figsize=(8,5))
plt.bar(models, prices, color=["blue", "green", "red"])
plt.title("Option Price Comparison")
plt.grid()
plt.show()


# =========================================================
# 2. HESTON PRICE PATHS (FIXED)
# =========================================================
plt.figure(figsize=(12,5))
for i in range(50):
    plt.plot(S_path[i], alpha=0.3)

plt.title("Heston Price Paths (Correct Fixed Version)")
plt.grid()
plt.show()


# =========================================================
# 3. VOLATILITY PATHS
# =========================================================
plt.figure(figsize=(12,5))
for i in range(50):
    plt.plot(np.sqrt(v_path[i]), alpha=0.3)

plt.title("Heston Volatility Paths")
plt.grid()
plt.show()


# =========================================================
# 4. VOLATILITY SENSITIVITY
# =========================================================
vols = np.linspace(0.1, 0.5, 15)
bs_prices = []
heston_prices = []

for v in vols:
    bs_prices.append(black_scholes(S, K, T, r, v))

    hp, _, _ = heston_mc(
        S, K, T, r,
        v0=0.04, kappa=2, theta=0.04,
        xi=0.3, rho=-0.7,
        steps=100, paths=2000
    )
    heston_prices.append(hp)

plt.figure(figsize=(10,5))
plt.plot(vols, bs_prices, label="Black-Scholes")
plt.plot(vols, heston_prices, label="Heston MC")
plt.title("Volatility Sensitivity Comparison")
plt.legend()
plt.grid()
plt.show()