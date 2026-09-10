# First exercise

from numpy import arange
from timeit import timeit
from numpy import dot
from numpy import polyval as polyval_proper

# ================ First exercise ==================   
def explicit_loop(coefficients, rate):
    final_res = 0.0

    for i, coefficient in enumerate(coefficients, start=1):
        final_res += coefficient / (1 + rate) ** i

    return final_res


def horner(coefficients, rate):
    q = 1 / (1 + rate)
    final_res = 0.0

    for coefficient in reversed(coefficients):
        final_res = (final_res + coefficient) * q

    return final_res


def polyval(coefficients, rate):
    q = 1 / (1 + rate)

    return q * polyval_proper(coefficients[::-1], q)


def dot_prod(coefficients, rate):
    q = 1 / (1 + rate)
    list_of_q = []

    for i in range(1, len(coefficients) + 1):
        list_of_q.append(q ** i)

    final_res = dot(coefficients, list_of_q)

    return final_res

def sol_first_ex(C, r):
    present_value = explicit_loop(C, r)

    t1 = timeit(lambda: explicit_loop(C, r), number=1000)
    t2 = timeit(lambda: horner(C, r), number=1000)
    t3 = timeit(lambda: polyval(C, r), number=1000)
    t4 = timeit(lambda: dot_prod(C, r), number=1000)

    print(f"Present value: {present_value:.2f}")
    print(f"Explicit loop time: {t1}")
    print(f"Horner's Scheme time: {t2}")
    print(f"Polyval function time: {t3}")
    print(f"Dot Product time: {t4}")

# =========== second exercise ==============
def sol_second_ex(rate, put_months, take_months):
    # retire in 40 years , amount A in the bank each month for 480 months, after which she will withdraw 2000 for 360 months
    # rate is 0.02 anually

    i = rate/12 

    q_monthly = 1 / (1 + i)

    # PV_withdrawls = FV_savings => A*(1 + i)*((1+i)**480 - 1)/i = 2000*(1 + i)*((1 - (1 + i)**-360)/i) => the following value of A:

    A = 2000*(1 - (1 + i) ** (-take_months))/((1 + i) ** put_months - 1)

    print(f"the value that has to be introduced at the beginning of each month for 480 should be {A}")

# ========= third exercise =========

def sol_third_ex(P, r, m, n):
    period_rate = r / m
    total_payments = m * n

    monthly_payment = P * period_rate / (
        1 - (1 + period_rate) ** (-total_payments)
    )

    effective_annual_rate = (1 + period_rate) ** m - 1

    print(f"Monthly payment: {monthly_payment:.2f}")
    print(f"Effective annual interest rate: {effective_annual_rate * 100:.4f}%")
    print()

    print(
        f"{'Month':>6} "
        f"{'Interest':>12} "
        f"{'Principal':>12} "
        f"{'Remaining principal':>20}"
    )

    remaining_principal = P

    for month in range(1, total_payments + 1):
        interest_part = remaining_principal * period_rate
        principal_part = monthly_payment - interest_part

        remaining_principal -= principal_part

        if abs(remaining_principal) < 1e-8:
            remaining_principal = 0.0

        print(
            f"{month:6d} "
            f"{interest_part:12.2f} "
            f"{principal_part:12.2f} "
            f"{remaining_principal:20.2f}"
        )

# ========= fourth exercise =========

from scipy.optimize import brentq


def sol_fourth_ex(C, P):
    def irr_equation(rate):
        present_value = 0.0

        for i, cash_flow in enumerate(C, start=1):
            present_value += cash_flow / (1 + rate) ** i

        return present_value - P

    irr = brentq(irr_equation, 0.0, 0.1)

    print(f"IRR: {irr:.6f}")
    print(f"IRR percentage: {irr * 100:.4f}%")


def main(): # Uncomment each solution one by one so it is easier to see the output 

    sol_first_ex(120.0 * arange(500, 1200), 0.01)
    #sol_second_ex(0.02, 480, 360)
    # sol_third_ex(500000, 0.02, 12, 20)
    #sol_fourth_ex(120.0 * arange(42, 52), 50000.0)

if __name__ == "__main__":
    main()