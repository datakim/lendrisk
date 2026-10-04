import pandas as pd
import pytest

from lendrisk import RevenueAdvance, RevenueShock, compare_scenarios, make_merchant_cashflows


def test_synthetic_data_is_reproducible_and_scaled():
    a = make_merchant_cashflows(days=30, seed=17)
    b = make_merchant_cashflows(days=30, seed=17)
    c = make_merchant_cashflows(days=30, seed=17, daily_revenue=2000)
    pd.testing.assert_frame_equal(a, b)
    pd.testing.assert_series_equal(a["revenue"] * 2, c["revenue"])


def test_shock_preserves_input_and_fixed_costs(daily_path):
    frame = daily_path()
    original = frame.copy(deep=True)
    shocked = RevenueShock("down", 0.8, start_date="2025-01-03", end_date="2025-01-04").apply(frame)
    pd.testing.assert_frame_equal(frame, original)
    assert shocked["revenue"].tolist() == [100, 100, 80, 80, 100, 100, 100, 100, 100, 100]
    pd.testing.assert_series_equal(frame["operating_cost"], shocked["operating_cost"])


def test_cost_elasticity(daily_path):
    shocked = RevenueShock("down", 0.8, cost_elasticity=0.5).apply(daily_path())
    assert shocked["operating_cost"].iloc[0] == pytest.approx(54)


def test_downside_reveals_horizon_censoring(daily_path):
    report = compare_scenarios(
        RevenueAdvance(100, 1.2, 0.2), daily_path(days=10), [RevenueShock("down", 0.4)]
    )
    assert report.loc["baseline", "repaid"]
    assert not report.loc["down", "repaid"]
    assert report.loc["down", "remaining_balance"] == pytest.approx(40)
    assert pd.isna(report.loc["down", "payoff_days"])


def test_scenario_names_cannot_replace_baseline(daily_path):
    with pytest.raises(ValueError, match="unique"):
        compare_scenarios(
            RevenueAdvance(100, 1.2, 0.2), daily_path(), [RevenueShock("baseline", 0.8)]
        )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"revenue_multiplier": -1},
        {"cost_elasticity": 2},
        {"start_date": "2025-02-01", "end_date": "2025-01-01"},
    ],
)
def test_invalid_scenarios(kwargs):
    with pytest.raises(ValueError):
        RevenueShock("bad", **{"revenue_multiplier": 0.8, **kwargs})
