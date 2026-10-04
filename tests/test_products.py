from decimal import Decimal, localcontext

import numpy as np
import numpy_financial as npf
import pandas as pd
import pytest

from lendrisk import MinimumPayment, RepaymentMilestone, RevenueAdvance, TermLoan


def test_closed_form_payoff_and_no_overcollection(daily_path):
    result = RevenueAdvance(100, 1.2, 0.2).simulate(daily_path(days=10))
    report = result.summary()
    assert report["payoff_days"] == 6
    assert report["total_paid"] == pytest.approx(120)
    assert report["remaining_balance"] == 0
    assert result.schedule["payment"].iloc[6:].eq(0).all()
    assert len(result.schedule) == 10


def test_partial_last_payment_is_capped(daily_path):
    result = RevenueAdvance(105, 1, 0.2).simulate(daily_path())
    assert result.schedule["payment"].iloc[5] == pytest.approx(5)
    assert result.summary()["total_paid"] == pytest.approx(105)


def test_zero_sales_preserves_balance_and_unknown_return(daily_path):
    result = RevenueAdvance(100, 1.2, 0.2).simulate(daily_path(revenue=0))
    report = result.summary()
    assert not report["repaid"]
    assert report["payoff_days"] is None
    assert report["effective_annual_return"] is None
    assert report["remaining_balance"] == 120
    assert report["recovery_ratio"] == 0


def test_cash_conservation_and_negative_cash_is_not_default(daily_path):
    path = daily_path(days=10, revenue=10, cost=100)
    result = RevenueAdvance(100, 1.2, 0.5).simulate(path, opening_cash=20)
    expected = (
        120 + (path["revenue"] - path["operating_cost"] - result.schedule["payment"]).cumsum()
    )
    np.testing.assert_allclose(result.schedule["cash_balance"], expected)
    assert result.summary()["negative_cash_days"] > 0
    assert result.summary()["total_paid"] == 50


def test_cash_unknown_without_costs(daily_path):
    result = RevenueAdvance(100, 1.2, 0.2).simulate(daily_path(cost=None))
    assert result.summary()["minimum_cash_balance"] is None
    assert result.schedule["cash_balance"].isna().all()


def test_floor_is_block_topup_not_an_extra_daily_payment(daily_path):
    product = RevenueAdvance(100, 1.2, 0.1, minimum_payment=MinimumPayment(30, 3))
    result = product.simulate(daily_path(days=7, revenue=10))
    np.testing.assert_allclose(result.schedule["minimum_topup"], [0, 0, 27, 0, 0, 27, 0])
    assert result.summary()["total_paid"] == 61
    assert result.summary()["remaining_balance"] == 59


def test_floor_cannot_collect_more_than_balance(daily_path):
    product = RevenueAdvance(100, 1.2, 0.1, minimum_payment=MinimumPayment(1_000, 3))
    result = product.simulate(daily_path(days=7, revenue=0))
    assert result.summary()["total_paid"] == 120
    assert result.schedule["minimum_topup"].iloc[2] == 120
    assert result.schedule["payment"].iloc[3:].eq(0).all()


def test_milestones_check_without_creating_payments(daily_path):
    product = RevenueAdvance(
        100,
        1.2,
        0.1,
        milestones=(
            RepaymentMilestone(3, 0.5),
            RepaymentMilestone(20, 1.0),
        ),
    )
    result = product.simulate(daily_path(days=7, revenue=10))
    assert result.summary()["total_paid"] == 7
    assert result.summary()["milestones_observed"] == 1
    assert result.summary()["milestones_breached"] == 1
    assert result.schedule["milestone_shortfall"].iloc[2] == 57


def test_different_holdbacks_change_timing_not_total_fee(daily_path):
    path = daily_path(days=200)
    slower = RevenueAdvance(100, 1.2, 0.05).simulate(path).summary()
    faster = RevenueAdvance(100, 1.2, 0.10).simulate(path).summary()
    assert faster["payoff_days"] < slower["payoff_days"]
    assert faster["total_paid"] == pytest.approx(slower["total_paid"])
    assert faster["effective_annual_return"] > slower["effective_annual_return"]


@pytest.mark.parametrize("seed", range(5))
def test_random_path_payment_and_cash_invariants(seed):
    rng = np.random.default_rng(seed)
    path = pd.DataFrame(
        {
            "date": pd.date_range("2025-01-01", periods=120),
            "revenue": rng.uniform(0, 1000, 120),
            "operating_cost": rng.uniform(0, 500, 120),
        }
    )
    result = RevenueAdvance(5_000, 1.15, 0.2).simulate(path, opening_cash=100)
    schedule = result.schedule
    assert schedule["payment"].ge(0).all()
    assert (schedule["payment"] <= schedule["revenue"] * 0.2 + 1e-8).all()
    assert schedule["remaining_balance"].diff().dropna().le(0).all()
    np.testing.assert_allclose(schedule["cumulative_paid"] + schedule["remaining_balance"], 5750)
    assert schedule["cash_balance"].iloc[-1] == pytest.approx(
        5100 + (path["revenue"] - path["operating_cost"]).sum() - schedule["payment"].sum()
    )


@pytest.mark.parametrize(
    "principal,rate,months",
    [(10000, 0, 12), (10000, 0.12, 12), (200000, 0.075, 180)],
)
def test_term_payment_against_numpy_financial(principal, rate, months):
    loan = TermLoan(principal, rate, months)
    assert loan.monthly_payment == pytest.approx(-npf.pmt(rate / 12, months, principal), rel=1e-10)
    schedule = loan.schedule(start_date="2025-01-31")
    assert schedule["principal_paid"].sum() == pytest.approx(principal)
    assert schedule["remaining_balance"].iloc[-1] == 0


def test_near_zero_interest_against_high_precision_reference():
    # The usual float (1+r)**n - 1 expression loses precision for tiny r.
    with localcontext() as context:
        context.prec = 60
        rate = Decimal("1e-12") / 12
        growth = (1 + rate) ** 24
        reference = Decimal(1000) * rate * growth / (growth - 1)
    loan = TermLoan(1000, 1e-12, 24)
    assert loan.monthly_payment == pytest.approx(float(reference), rel=1e-13)


def test_term_due_dates_anchor_to_funding_date():
    result = TermLoan(300, 0, 3).schedule(start_date="2025-01-31")
    assert result["date"].tolist() == list(
        pd.to_datetime(["2025-02-28", "2025-03-31", "2025-04-30"])
    )
    assert result["payment"].tolist() == [100, 100, 100]


@pytest.mark.parametrize(
    "args", [(0, 1.2, 0.1), (100, 0.9, 0.1), (100, 1.2, 0), (100, 1.2, 1.1), (np.inf, 1.2, 0.1)]
)
def test_invalid_advance_contract(args):
    with pytest.raises(ValueError):
        RevenueAdvance(*args)


def test_decreasing_milestones_rejected():
    with pytest.raises(ValueError, match="not decrease"):
        RevenueAdvance(
            100,
            1.1,
            0.1,
            milestones=(
                RepaymentMilestone(10, 0.8),
                RepaymentMilestone(20, 0.5),
            ),
        )


def test_incomplete_daily_path_cannot_skip_payment_dates(daily_path):
    with pytest.raises(ValueError, match="missing calendar"):
        RevenueAdvance(100, 1.2, 0.1).simulate(daily_path().drop(index=3))
