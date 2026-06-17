import numpy as np
import matplotlib.pyplot as plt

# =========================
# BINOMIAL PRICING ENGINE
# =========================
def binomial_option_price(S, K, T, r, sigma, steps=100, option_type="call"):
    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1 / u
    p = (np.exp(r * dt) - d) / (u - d)

    # stock tree
    stock = np.zeros((steps + 1, steps + 1))
    stock[0, 0] = S

    for i in range(1, steps + 1):
        stock[i, 0] = stock[i - 1, 0] * u
        for j in range(1, i + 1):
            stock[i, j] = stock[i - 1, j - 1] * d

    # option value at maturity
    option = np.zeros((steps + 1, steps + 1))

    for j in range(steps + 1):
        if option_type == "call":
            option[steps, j] = max(0, stock[steps, j] - K)
        else:
            option[steps, j] = max(0, K - stock[steps, j])

    # backward induction
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            option[i, j] = np.exp(-r * dt) * (
                p * option[i + 1, j] + (1 - p) * option[i + 1, j + 1]
            )

    return option[0, 0], stock, option, p


# =========================
# PARAMETERS
# =========================
S0 = 100
K = 100
T = 1
r = 0.05
sigma = 0.2


# =========================
# PRICING
# =========================
call_price, stock_tree, option_tree, p = binomial_option_price(
    S0, K, T, r, sigma, steps=100, option_type="call"
)

put_price, _, _, _ = binomial_option_price(
    S0, K, T, r, sigma, steps=100, option_type="put"
)

print("\n==============================")
print(" BINOMIAL OPTION PRICING ")
print("==============================")

print(f"Call Price: {call_price:.4f}")
print(f"Put Price:  {put_price:.4f}")
print(f"Risk-neutral probability: {p:.4f}")


# =========================
# CONVERGENCE TEST
# =========================
steps_range = [10, 25, 50, 100, 200]
prices = []

for n in steps_range:
    price, _, _, _ = binomial_option_price(S0, K, T, r, sigma, steps=n)
    prices.append(price)

plt.figure(figsize=(10,5))
plt.plot(steps_range, prices, marker="o")
plt.title("Binomial Convergence to Stable Price")
plt.xlabel("Steps")
plt.ylabel("Option Price")
plt.grid()
plt.show()


# =========================
# PAYOFF DIAGRAM
# =========================
S_range = np.linspace(50, 150, 200)

call_payoff = np.maximum(S_range - K, 0) - call_price
put_payoff = np.maximum(K - S_range, 0) - put_price

plt.figure(figsize=(10,5))
plt.plot(S_range, call_payoff, label="Call PnL")
plt.plot(S_range, put_payoff, label="Put PnL")
plt.axhline(0, linestyle="--", color="black")
plt.title("Option Payoff at Expiry (Binomial Model)")
plt.legend()
plt.grid()
plt.show()


# =========================
# STOCK TREE VISUAL (SMALL SAMPLE)
# =========================
def plot_tree(tree, title):
    plt.figure(figsize=(10,5))
    for i in range(10):  # only first 10 paths for clarity
        plt.plot(tree[:, i], alpha=0.6)
    plt.title(title)
    plt.grid()
    plt.show()

plot_tree(stock_tree, "Stock Price Tree (Sample Paths)")


# =========================
# OPTION TREE VISUAL
# =========================
plot_tree(option_tree, "Option Value Tree (Backward Induction View)")


# =========================
# IMPLIED DYNAMICS: DELTA APPROX
# =========================
def binomial_delta(S, K, T, r, sigma):
    up = binomial_option_price(S*1.01, K, T, r, sigma)[0]
    down = binomial_option_price(S*0.99, K, T, r, sigma)[0]
    return (up - down) / (0.02 * S)

delta = binomial_delta(S0, K, T, r, sigma)

print(f"\nApprox Delta (finite diff): {delta:.4f}")