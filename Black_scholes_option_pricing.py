import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

# =========================
# BLACK-SCHOLES PRICING
# =========================
def black_scholes(S, K, T, r, sigma, option_type="call"):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    if option_type == "call":
        price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    else:
        price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

    return price, d1, d2


# =========================
# GREEKS
# =========================
def greeks(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    delta = norm.cdf(d1)
    gamma = norm.pdf(d1) / (S * sigma * np.sqrt(T))
    vega = S * norm.pdf(d1) * np.sqrt(T) / 100
    theta = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T)) / 365
    rho = K * T * np.exp(-r * T) * norm.cdf(d2) / 100

    return delta, gamma, vega, theta, rho


# =========================
# PARAMETERS
# =========================
S0 = 100
K = 100
T = 1.0
r = 0.05
sigma = 0.2


# =========================
# PRICING
# =========================
call_price, _, _ = black_scholes(S0, K, T, r, sigma, "call")
put_price, _, _ = black_scholes(S0, K, T, r, sigma, "put")

delta, gamma, vega, theta, rho = greeks(S0, K, T, r, sigma)


print("\n==============================")
print(" BLACK-SCHOLES ENGINE REPORT ")
print("==============================")

print(f"Call Price: {call_price:.4f}")
print(f"Put Price:  {put_price:.4f}")

print("\n--- GREEKS ---")
print(f"Delta: {delta:.4f}")
print(f"Gamma: {gamma:.6f}")
print(f"Vega:  {vega:.4f}")
print(f"Theta: {theta:.6f}")
print(f"Rho:   {rho:.4f}")


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
plt.title("Option Payoff at Expiry")
plt.legend()
plt.grid()
plt.show()


# =========================
# DELTA PROFILE
# =========================
deltas = []
spots = np.linspace(50, 150, 200)

for s in spots:
    d, _, _, _, _ = greeks(s, K, T, r, sigma)
    deltas.append(d)

plt.figure(figsize=(10,5))
plt.plot(spots, deltas)
plt.title("Delta vs Spot Price")
plt.xlabel("Spot Price")
plt.ylabel("Delta")
plt.grid()
plt.show()


# =========================
# VEGA SENSITIVITY
# =========================
vols = np.linspace(0.05, 0.6, 50)
prices = []

for v in vols:
    price, _, _ = black_scholes(S0, K, T, r, v, "call")
    prices.append(price)

plt.figure(figsize=(10,5))
plt.plot(vols, prices)
plt.title("Call Price vs Volatility")
plt.xlabel("Volatility")
plt.ylabel("Price")
plt.grid()
plt.show()


# =========================
# TIME DECAY CURVE
# =========================
times = np.linspace(0.01, 2, 100)
time_prices = []

for t in times:
    price, _, _ = black_scholes(S0, K, t, r, sigma, "call")
    time_prices.append(price)

plt.figure(figsize=(10,5))
plt.plot(times, time_prices)
plt.title("Call Price vs Time to Maturity")
plt.xlabel("Time")
plt.ylabel("Price")
plt.grid()
plt.show()