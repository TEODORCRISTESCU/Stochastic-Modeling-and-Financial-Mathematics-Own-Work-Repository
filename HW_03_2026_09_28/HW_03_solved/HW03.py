from pathlib import Path
import re

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ===== Problem 1 ====

data_path = Path(__file__).with_name("data.csv")

if data_path.exists():
    df = pd.read_csv(data_path)

    spot_rates = df[df["KEY"].str.contains(r"\.SR_", na=False)].copy()
    spot_rates["MATURITY_CODE"] = (
        spot_rates["KEY"].str.extract(r"\.SR_(\d+Y\d*M|\d+Y|\d+M)$")[0]
    )

    def maturity_in_years(code):
        # convert ECB maturity labels such as 6M, 2Y or 10Y6M into years. 
        if pd.isna(code):
            return np.nan
        year_match = re.search(r"(\d+)Y", code)
        month_match = re.search(r"(\d+)M", code)
        years = int(year_match.group(1)) if year_match else 0
        months = int(month_match.group(1)) if month_match else 0
        return years + months / 12

    spot_rates["MATURITY"] = spot_rates["MATURITY_CODE"].apply(maturity_in_years)
    spot_rates["TIME_PERIOD"] = pd.to_datetime(spot_rates["TIME_PERIOD"], errors="coerce")
    spot_rates["OBS_VALUE"] = pd.to_numeric(spot_rates["OBS_VALUE"], errors="coerce")
    spot_rates = spot_rates.dropna(subset=["MATURITY", "TIME_PERIOD", "OBS_VALUE"])

    if spot_rates.empty:
        print("Problem 1: No usable spot-rate observations found in data.csv.")
    else:
        latest_date = spot_rates["TIME_PERIOD"].max()
        latest_rates = spot_rates.loc[
            spot_rates["TIME_PERIOD"] == latest_date
        ].sort_values("MATURITY")

        print("PROBLEM 1: ECB SPOT-RATE DATA")
        print(f"Latest observation date in file: {latest_date.date()}")
        print(f"Number of maturity observations: {len(latest_rates)}")
        print("\nThe ECB file contains time-stamped financial series, including")
        print("spot rates, forward rates and Svensson yield-curve parameters.")
        print("KEY identifies the series, TIME_PERIOD gives the date, and")
        print("OBS_VALUE gives its observed value. Spot rates are annualized")
        print("percentages for different maturities. The curve below uses")
        print("the latest date available in the local CSV file.\n")

        plt.figure(figsize=(9, 5))
        plt.plot(latest_rates["MATURITY"], latest_rates["OBS_VALUE"], marker="o")
        plt.title(f"ECB Euro Area Spot Yield Curve ({latest_date.date()})")
        plt.xlabel("Maturity (years)")
        plt.ylabel("Spot rate (% per annum)")
        plt.grid(True, alpha=0.35)
        plt.tight_layout()
        plt.savefig(Path(__file__).with_name("problem1_yield_curve.png"), dpi=150)
        plt.show()
else:
    print(f"Problem 1: Cannot find {data_path.name} next to this Python file.")
    print("Place the downloaded ECB CSV there to generate the yield curve.\n")

# ===== Problem 2 ======

def call_payoff(stock_price, strike):
    return np.maximum(stock_price - strike, 0)


def put_payoff(stock_price, strike):
    return np.maximum(strike - stock_price, 0)


stock_prices = np.linspace(0, 200, 401)
strike = 100

plt.figure(figsize=(9, 5))
plt.plot(stock_prices, call_payoff(stock_prices, strike), label="European call")
plt.plot(stock_prices, put_payoff(stock_prices, strike), label="European put")
plt.axvline(strike, color="gray", linestyle="--", alpha=0.6, label="Strike = $100")
plt.title("European Option Payoffs at Expiration")
plt.xlabel("Stock price at expiration ($)")
plt.ylabel("Payoff ($)")
plt.legend()
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(Path(__file__).with_name("problem2_option_payoffs.png"), dpi=150)
plt.show()

# ====Problem 3 ===

stock_prices = np.linspace(30, 110, 401)
butterfly_payoff = (
    call_payoff(stock_prices, 50)
    - 2 * call_payoff(stock_prices, 70)
    + call_payoff(stock_prices, 90)
)

print("PROBLEM 3: BUTTERFLY SPREAD")
print("Buy one call at K = $50; sell two calls at K = $70;")
print("buy one call at K = $90 (all with the same expiration).")
print("Payoff = max(S-50, 0) - 2 max(S-70, 0) + max(S-90, 0).\n")

plt.figure(figsize=(9, 5))
plt.plot(stock_prices, butterfly_payoff)
plt.title("Butterfly Spread")
plt.xlabel("Stock price at expiration ($)")
plt.ylabel("Payoff ($)")
plt.xlim(30, 110)
plt.ylim(-1, 22)
plt.grid(True, alpha=0.35)
plt.tight_layout()
plt.savefig(Path(__file__).with_name("problem3_butterfly_spread.png"), dpi=150)
plt.show()

# ==== Problem 4 =====

print("PROBLEM 4: THREE-STATE STOCK MODEL")
print(
    "In the one-period binomial model, a portfolio of stock and a risk-free asset has two unknown holdings."
    "The payoff can be matched in both future states by solving two equations. "
    "If there are three possible future stock prices, matching an arbitrary option payoff requires three equations but "
    "there are still only two holdings. Usually no exact "
    "replicating portfolio exists, so the market is incomplete "
    "and absence of arbitrage does not generally determine "
    "one unique option price. Replication can still work for "
    "certain particular payoffs, or if a suitable third traded "
    "security is introduced."
)
