import numpy as np
from timeit import timeit
from scipy.optimize import brentq
import matplotlib.pyplot as plt

from scipy.optimize import brent
# ==== First exercise =====

N = 300

C = 120.0 * np.arange(10, N + 10)

P = 15000

def irr_f(r):
    PV = 0 
    i = 1
    for cashflow in C:
        PV += cashflow/ (1 + r) ** i
        i += 1

    PV -= P

    return PV

def bisection(f, tol = 1e-8 ):
    a = 0.0
    b = 0.1

    while f(a) *f(b) > 0:
        b += 0.1

    if f(a) * f(b) > 0:
        raise ValueError("Multiplication sign stayed the same")
    
    while (b - a) > tol:
        m = (a + b) / 2

        if f(m) == 0:
            return m

        if f(m) * f(a) < 0:
            b = m
        
        else:
            a = m
    
    return (a + b) / 2


def irr_df(r):

    deriv = 0 
    for i, cashflow in enumerate(C, start = 1):
        deriv += - i * cashflow/ (1 + r) ** (i + 1)

    return deriv

def newton(f, df, r, tol = 1e-8):
    
    while abs(f(r)) > tol:
        r = r - f(r)/df(r)
    
    return r

def secant(f, x0, x1, tol=1e-10):

    while abs(f(x1)) > tol:

        denominator = f(x1) - f(x0)

        if denominator == 0:
            raise ValueError("Division by zero in secant method")

        x_new = (
            x1
            - f(x1) * (x1 - x0) / denominator
        )

        x0 = x1
        x1 = x_new

    return x1

def bond_yield_f(i):
    P = 75 
    F = 100
    C = 0.10 * 100 / 2
    N = 20
    result = 0
    for j in range(1, N + 1):
        result += C/(1 + i) ** j
    
    result +=  F/(1 + i) ** j

    return result - P

def second_ex():

    i = brentq(bond_yield_f, 0.0, 0.2)

    print(f"Yield of maturity of bond is {i * 2}")
    
def bond_price(par, coupon_rate, yield_rate, maturity, m=2):
    coupon = coupon_rate * par / m
    period_yield = yield_rate / m
    N = int(m * maturity)

    j = np.arange(1, N + 1)

    coupon_pv = np.sum(
        coupon / (1 + period_yield) ** j
    )

    par_pv = par / (1 + period_yield) ** N

    return coupon_pv + par_pv

def third_ex():
    maturities = np.arange(1, 31)

    prices_2 = []
    prices_6 = []
    prices_12 = []

    for T in maturities:
        prices_2.append(bond_price(1000, 0.02, 0.06, T))
        prices_6.append(bond_price(1000, 0.06, 0.06, T))
        prices_12.append(bond_price(1000, 0.12, 0.06, T))

    plt.plot(maturities, prices_2, label="2% coupon")
    plt.plot(maturities, prices_6, label="6% coupon")
    plt.plot(maturities, prices_12, label="12% coupon")

    plt.xlabel("Time to maturity (years)")  
    plt.ylabel("Bond price ($)")
    plt.legend()
    plt.grid()

    plt.show()

def fourth_ex():
    N = 10
    F = 1000
    coupon_rate = 0.08
    Coupon = 0.08 * 1000
    def bond_price(r):
        j = np.arange(1, N + 1)

        coupon_pv = np.sum(
        Coupon / (1 + r) ** j
        )

        par_pv = F / (1 + r) ** N

        return coupon_pv + par_pv


    yields = np.linspace(0.01, 0.15, 100)

    prices = []

    for r in yields:
        prices.append(bond_price(r))

    plt.plot(yields * 100, prices)

    plt.xlabel("Yield (%)")
    plt.ylabel("Bond price ($)")
    plt.grid()

    plt.show()

import numpy as np
import matplotlib.pyplot as plt


def fifth_ex():
    F = 1000.0
    yield_rate = 0.06
    m = 2

    coupon_rates = [0.02, 0.06, 0.12]

    maturities = np.arange(0, 100.5, 0.5)

    def bond_volatility(T, coupon_rate):
        if T == 0:
            return 0.0

        N = int(T * m)
        period_rate = yield_rate / m
        coupon = coupon_rate * F / m

        j = np.arange(1, N + 1)

        cashflows = np.full(N, coupon)
        cashflows[-1] += F

        P = np.sum(
            cashflows / (1 + period_rate) ** j
        )
        
        dP_dr = np.sum(
            -(j / m) * cashflows
            / (1 + period_rate) ** (j + 1)
        )

        return -dP_dr / P

    for coupon_rate in coupon_rates:
        volatilities = np.array([
            bond_volatility(T, coupon_rate)
            for T in maturities
        ])

        plt.plot(
            maturities,
            volatilities,
            label=f"{coupon_rate * 100:.0f}% coupon"
        )

    plt.xlabel("Time to maturity (years)")
    plt.ylabel("Price volatility")
    plt.title("Bond Price Volatility vs. Time to Maturity")
    plt.legend()
    plt.grid()

    plt.show()


def sixth_ex():
    F = 1000.0
    coupon_rate = 0.08
    m = 2
    T = 15

    coupon = coupon_rate * F / m


    def bond_price(rate):
        period_rate = rate / m
        N = T * m

        j = np.arange(1, N + 1)

        coupon_pv = np.sum( coupon / (1 + period_rate) ** j)

        par_pv = F / (1 + period_rate) ** N

        return coupon_pv + par_pv


    rates = [0.06, 0.08, 0.10]

    years_to_maturity = np.linspace(15, 0, 200)

    for r in rates:
        P0 = bond_price(r)

        years_elapsed = T - years_to_maturity

        forward_value = (P0 * (1 + r / m) ** (m * years_elapsed))

        plt.plot(
            years_to_maturity,
            forward_value,
            label=f"{r * 100:.0f}% rate"
        )

    plt.xlabel("Years to maturity")
    plt.ylabel("Forward value ($)")
    plt.legend()
    plt.grid()

    plt.show()

def seventh_ex():
    F = 1000.0
    coupon_rate = 0.10
    coupon = coupon_rate * F

    total_maturity = 30
    horizon = 10
    remaining_maturity = total_maturity - horizon


    def horizon_value(r):
        if abs(r) < 1e-12:
            fv_coupons = coupon * horizon
        else:
            fv_coupons = coupon * ((1 + r) ** horizon - 1) / r

            j = np.arange(1, remaining_maturity + 1)

            remaining_bond_value = (np.sum(coupon / (1 + r) ** j)+ F / (1 + r) ** remaining_maturity)

        return fv_coupons + remaining_bond_value

    yields = np.linspace(0.001, 0.30, 500)
    values = np.array([horizon_value(r) for r in yields])

    r_min = brent(horizon_value, brack=(0.05, 0.10, 0.20))
    value_min = horizon_value(r_min)

    print(f"Yield at minimum: {r_min * 100:.4f}%")
    print(f"Minimum horizon value: ${value_min:.2f}")

    plt.plot(yields * 100, values)
    plt.scatter(r_min * 100, value_min)

    plt.xlabel("Yield (%)")
    plt.ylabel("Future value after 10 years ($)")
    plt.title("30-Year 10% Coupon Bond: 10-Year Horizon Value")
    plt.grid()

    plt.show()
def main():

    irr = bisection(irr_f)

    print(f"The value of the IRR, with Bisection Method is {irr} or {irr*100} %")

    irr = newton(irr_f, irr_df, 0.1)

    print (f"The value of the IRR, with Newton Method is {irr} or {irr * 100} % ")

    irr = secant(irr_f, 0.0, 0.2)

    print (f"The value of the IRR, with Secant method is {irr} or {irr*100} %")

    irr = brentq(irr_f, 0.0, 0.2)

    print (f"The value of the IRR, with the brentq function, is {irr} or {irr * 100} %")
    repetitions = 1000

    t_bisection = timeit(
        lambda: bisection(irr_f),
        number=repetitions
    )

    t_newton = timeit(
        lambda: newton(irr_f, irr_df, 0.1),
        number=repetitions
    )

    t_secant = timeit(
        lambda: secant(irr_f, 0.0, 0.2),
        number=repetitions
    )

    t_brentq = timeit(
        lambda: brentq(irr_f, 0.0, 0.2),
        number=repetitions
    )

    print()
    print(f"Run times for {repetitions} evaluations:")
    print(f"Bisection: {t_bisection:.6f} s")
    print(f"Newton:    {t_newton:.6f} s")
    print(f"Secant:    {t_secant:.6f} s")
    print(f"Brentq:    {t_brentq:.6f} s")


    second_ex()

    third_ex()

    fourth_ex()

    fifth_ex()
    
    sixth_ex()

    seventh_ex()

if __name__ == "__main__":
    main()
