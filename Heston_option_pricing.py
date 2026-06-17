import numpy as np
import matplotlib.pyplot as plt

# =========================
# HESTON MODEL PARAMETERS
# =========================
S0 = 100
v0 = 0.04          # initial variance (vol^2)
K = 100
T = 1.0
r = 0.05

kappa = 2.0        # mean reversion speed
theta = 0.04       # long-term variance
xi = 0.3           # volatility of volatility (vol of vol)
rho = -0.7         # correlation (important for skew)

steps = 252
paths = 20000

dt = T / steps
np.random.seed(42)


# =========================
# HESTON SIMULATION ENGINE
# =========================
def simulate_heston():
    S = np.full(paths, S0, dtype=float)
    v = np.full(paths, v0, dtype=float)

    S_path = np.zeros((paths, steps + 1))
    v_path = np.zeros((paths, steps + 1))

    S_path[:, 0] = S
    v_path[:, 0] = v

    for t in range(1, steps + 1):

        z1 = np.random.normal(size=paths)
        z2 = np.random.normal(size=paths)

        # correlated Brownian motions
        w1 = z1
        w2 = rho * z1 + np.sqrt(1 - rho**2) * z2

        # variance process (CIR process)
        v = np.abs(
            v + kappa * (theta - v) * dt + xi * np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * w2
        )

        # stock process
        S = S * np.exp(
            (r - 0.5 * v) * dt + np.sqrt(np.maximum(v, 0)) * np.sqrt(dt) * w1
        )

        S_path[:, t] = S
        v_path[:, t] = v

    return S_path, v_path


S_path, v_path = simulate_heston()


# =========================
# OPTION PRICING (MONTE CARLO)
# =========================
payoff_call = np.maximum(S_path[:, -1] - K, 0)
call_price = np.exp(-r * T) * np.mean(payoff_call)

payoff_put = np.maximum(K - S_path[:, -1], 0)
put_price = np.exp(-r * T) * np.mean(payoff_put)


# =========================
# RISK METRICS
# =========================
returns = S_path[:, -1] / S0 - 1

var_5 = np.percentile(returns, 5)
cvar_5 = returns[returns <= var_5].mean()

vol_avg = np.mean(np.sqrt(v_path[:, -1]))

print("\n==============================")
print(" HESTON MODEL REPORT ")
print("==============================")

print(f"Call Price (MC):     {call_price:.4f}")
print(f"Put Price (MC):      {put_price:.4f}")

print("\n--- RISK ---")
print(f"VaR (5%):            {var_5:.2%}")
print(f"CVaR (5%):           {cvar_5:.2%}")
print(f"Avg Terminal Vol:    {vol_avg:.2%}")


# =========================
# 1. SAMPLE PRICE PATHS
# =========================
plt.figure(figsize=(12,5))
for i in range(50):
    plt.plot(S_path[i], alpha=0.4)

plt.title("Heston Model: Sample Price Paths")
plt.grid()
plt.show()


# =========================
# 2. VOLATILITY PATHS
# =========================
plt.figure(figsize=(12,5))
for i in range(50):
    plt.plot(np.sqrt(v_path[i]), alpha=0.4)

plt.title("Stochastic Volatility Paths (Heston)")
plt.grid()
plt.show()


# =========================
# 3. FINAL PRICE DISTRIBUTION
# =========================
plt.figure(figsize=(10,5))
plt.hist(S_path[:, -1], bins=80, alpha=0.75)
plt.title("Terminal Stock Price Distribution")
plt.grid()
plt.show()


# =========================
# 4. IMPLIED VOL EFFECT (SMILE INTUITION)
# =========================
strikes = np.linspace(60, 140, 20)
prices = []

for k in strikes:
    payoff = np.maximum(S_path[:, -1] - k, 0)
    prices.append(np.exp(-r * T) * np.mean(payoff))

plt.figure(figsize=(10,5))
plt.plot(strikes, prices)
plt.title("Heston: Option Price vs Strike (Smile Effect)")
plt.xlabel("Strike")
plt.ylabel("Call Price")
plt.grid()
plt.show()


# =========================
# 5. VOL vs PRICE RELATION
# =========================
terminal_vol = np.sqrt(v_path[:, -1])

plt.figure(figsize=(8,6))
plt.scatter(terminal_vol, S_path[:, -1], alpha=0.2)
plt.title("Volatility vs Terminal Price")
plt.xlabel("Volatility")
plt.ylabel("Price")
plt.grid()
plt.show()