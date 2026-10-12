from math import erf, exp, log, sqrt
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np


# ===== Problem 1 =====


def call_payoff(stock_price, strike):
    return np.maximum(stock_price - strike, 0.0)


def put_payoff(stock_price, strike):
    return np.maximum(strike - stock_price, 0.0)


def binomial_tree(payoff, n, rp, sigma, S, K, T):
    dt = T / n
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    risk_free_growth = (1 + rp) ** dt
    q = (risk_free_growth - d) / (u - d)

    up_moves = np.arange(n + 1)
    stock_prices = S * u**up_moves * d ** (n - up_moves)
    option_values = payoff(stock_prices, K)

    for _ in range(n, 0, -1):
        option_values = (
            q * option_values[1:] + (1 - q) * option_values[:-1]
        ) / risk_free_growth

    return float(option_values[0])


def normal_cdf(x):
    return 0.5 * (1 + erf(x / sqrt(2)))


def black_scholes_call(S, K, rp, sigma, T):
    r = log(1 + rp)
    d1 = (log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)
    return S * normal_cdf(d1) - K * exp(-r * T) * normal_cdf(d2)


n = 1000
rp = 0.02
sigma = 0.5
S = 1
K = 0.7
T = 1

tree_call_price = binomial_tree(call_payoff, n, rp, sigma, S, K, T)
closed_form_call_price = black_scholes_call(S, K, rp, sigma, T)

print("PROBLEM 1")
print(f"Binomial tree European call price: {tree_call_price:.6f}")
print(f"Closed-form European call price:   {closed_form_call_price:.6f}")
print(f"Absolute difference:               {abs(tree_call_price - closed_form_call_price):.6e}")
print()


# ===== Problem 2 =====


def option_tree(payoff, n, rp, sigma, S, K, T, american=False):
    dt = T / n
    u = exp(sigma * sqrt(dt))
    d = 1 / u
    risk_free_growth = (1 + rp) ** dt
    q = (risk_free_growth - d) / (u - d)

    up_moves = np.arange(n + 1)
    stock_prices = S * u**up_moves * d ** (n - up_moves)
    option_values = payoff(stock_prices, K)

    for step in range(n - 1, -1, -1):
        option_values = (
            q * option_values[1:] + (1 - q) * option_values[:-1]
        ) / risk_free_growth

        if american:
            up_moves = np.arange(step + 1)
            stock_prices = S * u**up_moves * d ** (step - up_moves)
            option_values = np.maximum(option_values, payoff(stock_prices, K))

    return float(option_values[0])


strike_prices = np.linspace(0.5, 1.5, 41)
put_parameters = {
    "n": 500,
    "rp": 0.04,
    "sigma": 0.35,
    "S": 1,
    "T": 1,
}

european_put_prices = np.array(
    [
        option_tree(put_payoff, K=strike, american=False, **put_parameters)
        for strike in strike_prices
    ]
)
american_put_prices = np.array(
    [
        option_tree(put_payoff, K=strike, american=True, **put_parameters)
        for strike in strike_prices
    ]
)

plt.figure(figsize=(9, 5))
plt.plot(strike_prices, european_put_prices, label="European put")
plt.plot(strike_prices, american_put_prices, label="American put", linestyle="--")
plt.title("European and American Put Prices")
plt.xlabel("Strike price K")
plt.ylabel("Option price at time 0")
plt.grid(True, alpha=0.35)
plt.legend()
plt.tight_layout()
plt.savefig(Path(__file__).with_name("problem2_put_prices.png"), dpi=150)
plt.close()

print("PROBLEM 2")
print("Saved plot: problem2_put_prices.png")
print(
    "The American put price is at least as high as the European put price, because the holder has all European exercise choices plus the extra choice to exercise early."
)
print(
    f"In this numerical example the maximum American premium is "
    f"{np.max(american_put_prices - european_put_prices):.6f}."
)
print()


# ===== Problem 3 =====


print("PROBLEM 3")
print(
    "Portfolio A has payoff C(T) - P(T) = max(S(T) - K, 0) - max(K - S(T), 0) = S(T) - K."
)
print(
    "Portfolio B, buying one stock and borrowing the present value of K due at time T, also has payoff S(T) - K."
)
print(
    "By no arbitrage, equal future payoffs must have equal prices today. Thus C0 - P0 = S0 - K / (1 + rp)^T, or equivalently C0 + K / (1 + rp)^T = P0 + S0."
)
