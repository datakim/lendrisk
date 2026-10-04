import numpy_financial as npf
import pytest

from lendrisk import xirr, xnpv


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
