import numpy as np
import matplotlib.pyplot as plt

# =========================
# CONFIG
# =========================
N_SIM = 20000
N_STEPS = 252
INITIAL = 10000

np.random.seed(42)

# =========================
# STRATEGY PARAMETERS
# =========================
def sample_params(n):
    win_rate = np.clip(np.random.normal(0.35, 0.05, n), 0.01, 0.99)
    payoff = np.clip(np.random.normal(2.0, 0.4, n), 0.5, 5.0)
    return win_rate, payoff

# =========================
# SIMULATION ENGINE
# =========================
def run_sim():
    eq = np.full((N_SIM, N_STEPS), INITIAL, dtype=float)
    peak = np.full(N_SIM, INITIAL)

    wr, pay = sample_params(N_SIM)

    for t in range(1, N_STEPS):
        prev = eq[:, t-1]
        peak = np.maximum(peak, prev)

        # dynamic risk scaling
        risk = 0.01 * np.where(prev < peak, 0.7, 1.1)

        wins = np.random.rand(N_SIM) < wr

        pnl = np.where(
            wins,
            risk * prev * pay,
            -risk * prev
        )

        eq[:, t] = prev + pnl

    return eq, wr, pay


eq, wr, pay = run_sim()

# =========================
# METRICS ENGINE
# =========================
final = eq[:, -1]
returns = final / INITIAL - 1

peak_curve = np.maximum.accumulate(eq, axis=1)
dd = (eq - peak_curve) / peak_curve
max_dd = dd.min(axis=1)

# recovery
recovery = []
never_recovered = 0
total_dd = 0

for i in range(N_SIM):
    path = eq[i]

    below = np.where(path < INITIAL)[0]
    above = np.where(path >= INITIAL)[0]

    if below.size == 0:
        continue

    total_dd += 1

    if above.size == 0:
        never_recovered += 1
        continue

    b = below[0]
    a = above[above > b]

    if a.size == 0:
        never_recovered += 1
        continue

    recovery.append(a[0] - b)

# =========================
# REPORT (INSTITUTIONAL LOG OUTPUT)
# =========================
mu = np.mean(returns)
sigma = np.std(returns)
sharpe = mu / (sigma + 1e-12)

var5 = np.percentile(returns, 5)
cvar5 = returns[returns <= var5].mean()

var1 = np.percentile(returns, 1)
cvar1 = returns[returns <= var1].mean()

ruin = np.mean(final < INITIAL * 0.8)

print("\n==============================")
print("   INSTITUTIONAL RISK REPORT  ")
print("==============================")

print(f"Simulations:            {N_SIM}")
print(f"Time Steps:             {N_STEPS}")

print("\n--- PERFORMANCE ---")
print(f"Mean Return:            {mu:.2%}")
print(f"Volatility:             {sigma:.2%}")
print(f"Sharpe Ratio:           {sharpe:.2f}")

print("\n--- TAIL RISK ---")
print(f"VaR (5%):               {var5:.2%}")
print(f"CVaR (5%):              {cvar5:.2%}")
print(f"VaR (1%):               {var1:.2%}")
print(f"CVaR (1%):              {cvar1:.2%}")

print("\n--- RISK ---")
print(f"Ruin Probability:       {ruin:.2%}")
print(f"Avg Max Drawdown:       {np.mean(max_dd):.2%}")

print("\n--- RECOVERY ---")
if recovery:
    print(f"Avg Recovery Time:      {np.mean(recovery):.2f} trades")
    print(f"Min Recovery Time:      {np.min(recovery)} trades")
else:
    print("Avg Recovery Time:      N/A")

if total_dd > 0:
    print(f"Non-Recovery Rate:      {never_recovered / total_dd:.2%}")


# =========================
# VISUALIZATION SUITE
# =========================

# 1. Equity Fan Chart
p5, p25, p50, p75, p95 = np.percentile(eq, [5, 25, 50, 75, 95], axis=0)

plt.figure(figsize=(12,6))
plt.fill_between(range(N_STEPS), p5, p95, alpha=0.2, label="5–95%")
plt.fill_between(range(N_STEPS), p25, p75, alpha=0.4, label="25–75%")
plt.plot(p50, linewidth=2, label="Median")
plt.title("Equity Curve Distribution (Monte Carlo Fan Chart)")
plt.legend()
plt.grid()
plt.show()

# 2. Final Returns Histogram
plt.figure(figsize=(10,5))
plt.hist(returns, bins=80, alpha=0.75)
plt.title("Final Return Distribution")
plt.grid()
plt.show()

# 3. Drawdown Histogram
plt.figure(figsize=(10,5))
plt.hist(max_dd, bins=80, alpha=0.75, color="red")
plt.title("Max Drawdown Distribution")
plt.grid()
plt.show()

# 4. Risk vs Return Scatter
plt.figure(figsize=(8,6))
plt.scatter(max_dd, returns, alpha=0.2)
plt.title("Risk vs Return")
plt.xlabel("Max Drawdown")
plt.ylabel("Return")
plt.grid()
plt.show()

# 5. Recovery Histogram
if recovery:
    plt.figure(figsize=(10,5))
    plt.hist(recovery, bins=60, alpha=0.7, color="green")
    plt.title("Recovery Time Distribution")
    plt.grid()
    plt.show()

# 6. Sample Paths
plt.figure(figsize=(12,6))
for i in range(50):
    plt.plot(eq[i], alpha=0.3)

plt.title("Sample Equity Paths")
plt.grid()
plt.show()
