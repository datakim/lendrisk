from decimal import Decimal, localcontext

import numpy_financial as npf
import pytest

from lendrisk import UnrepresentableReturnError, xirr, xnpv


def test_one_year_closed_form():
    dates = ["2025-01-01", "2026-01-01"]
    assert xirr([-100, 110], dates) == pytest.approx(0.1)
    assert xnpv(0.1, [-100, 110], dates) == pytest.approx(0, abs=1e-10)


def test_half_year_convention_and_negative_return():
    dates = ["2025-01-01", "2025-07-02"]  # 182 days
    assert xirr([-100, 95], dates) == pytest.approx(0.95 ** (365 / 182) - 1)


def test_equal_year_periods_match_numpy_financial():
    amounts = [-100, 60, 60]
    dates = ["2025-01-01", "2026-01-01", "2027-01-01"]
    assert xirr(amounts, dates) == pytest.approx(npf.irr(amounts))


def test_same_day_flows_are_netted_and_dates_sorted():
    assert xirr([110, -60, -40], ["2026-01-01", "2025-01-01", "2025-01-01"]) == pytest.approx(0.1)


def test_zero_first_cashflow_preserves_xnpv_origin():
    assert xnpv(0.1, [0, 110], ["2025-01-01", "2026-01-01"]) == pytest.approx(100)


def test_xirr_ignores_leading_zero():
    assert xirr([0, -100, 110], ["2024-01-01", "2025-01-01", "2026-01-01"]) == pytest.approx(0.1)


@pytest.mark.parametrize(
    "amounts,dates",
    [
        ([-100, 200, -50], ["2025-01-01", "2026-01-01", "2027-01-01"]),
        ([100, 110], ["2025-01-01", "2026-01-01"]),
        ([-100, 110], ["2025-01-01", "2025-01-01"]),
        ([-100, 110], ["2025-01-01"]),
        ([0, 0], ["2025-01-01", "2026-01-01"]),
    ],
)
def test_ambiguous_or_undefined_returns_rejected(amounts, dates):
    with pytest.raises(ValueError):
        xirr(amounts, dates)


def test_invalid_discount_rate():
    with pytest.raises(ValueError):
        xnpv(-1, [-100, 110], ["2025-01-01", "2026-01-01"])


def test_zero_cashflows_have_zero_present_value():
    assert xnpv(0.1, [0, 0], ["2025-01-01", "2026-01-01"]) == 0


def test_representable_high_return_is_not_limited_to_old_log_bracket():
    reference = float(5**365 - 1)
    assert xirr([-100, 500], ["2026-01-01", "2026-01-02"]) == pytest.approx(reference, rel=2e-12)


@pytest.mark.parametrize("initial,final", [("1e-250", "1e250"), ("1e250", "1e-250")])
def test_extreme_amounts_match_independent_decimal_closed_form(initial, final):
    # 36,524 actual days: long enough for intermediate discounts to overflow
    # or underflow in direct float arithmetic, while the return remains finite.
    with localcontext() as context:
        context.prec = 80
        reference = float((Decimal(final) / Decimal(initial)) ** (Decimal(365) / 36524) - 1)
    result = xirr([-float(initial), float(final)], ["1900-01-01", "2000-01-01"])
    assert result == pytest.approx(reference, rel=2e-12, abs=1e-15)


def test_unrepresentable_annual_return_raises_a_specific_value_error():
    with pytest.raises(UnrepresentableReturnError):
        xirr([-100, 10_000], ["2026-01-01", "2026-01-02"])


@pytest.mark.parametrize("rate,amount", [(100_000.0, "1e250"), (-0.99999, "1e-250")])
def test_xnpv_extreme_discount_preserves_representable_present_value(rate, amount):
    with localcontext() as context:
        context.prec = 80
        reference = float(
            Decimal(amount) * (1 + Decimal.from_float(rate)) ** (-Decimal(36524) / 365)
        )
    result = xnpv(rate, [0, float(amount)], ["1900-01-01", "2000-01-01"])
    assert result != 0
    assert result == pytest.approx(reference, rel=2e-12, abs=0)
