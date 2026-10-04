import numpy as np
import pandas as pd
import pytest

from lendrisk import cashflow_features, merchant_features


def test_known_complete_windows(daily_path):
    frame = daily_path(days=60, revenue=100, cost=60)
    values = cashflow_features(frame, windows=(30,))
    assert values["30d_revenue_sum"] == 3000
    assert values["30d_coverage"] == 1
    assert values["30d_revenue_cv"] == 0
    assert values["30d_revenue_growth"] == 0
    assert values["30d_operating_margin"] == pytest.approx(0.4)
    assert values["30d_top_day_share"] == pytest.approx(1 / 30)


def test_cutoff_blocks_future_values_and_invalid_future_revenue(daily_path):
    frame = daily_path(days=90)
    before = cashflow_features(frame.iloc[:60], as_of="2025-03-01", windows=(30,))
    frame.loc[60:, "revenue"] = np.nan
    after = cashflow_features(frame, as_of="2025-03-01", windows=(30,))
    pd.testing.assert_series_equal(before, after)


def test_partial_window_is_exposed(daily_path):
    values = cashflow_features(daily_path(days=10), windows=(30,))
    assert values["30d_observed_days"] == 10
    assert values["30d_coverage"] == pytest.approx(1 / 3)
    assert np.isnan(values["30d_revenue_growth"])


def test_no_observations_in_recent_window_is_not_zero_sales(daily_path):
    values = cashflow_features(daily_path(days=10), as_of="2025-06-01", windows=(30,))
    assert values["30d_coverage"] == 0
    assert np.isnan(values["30d_revenue_sum"])


def test_zero_revenue_ratios_remain_undefined(daily_path):
    values = cashflow_features(daily_path(days=60, revenue=0), windows=(30,))
    assert values["30d_zero_revenue_share"] == 1
    assert values["30d_negative_operating_cash_share"] == 1
    for name in ["revenue_cv", "top_day_share", "revenue_growth", "operating_margin"]:
        assert np.isnan(values["30d_" + name])


def test_growth_requires_two_full_calendar_windows(daily_path):
    frame = daily_path(days=60)
    frame.loc[30:, "revenue"] *= 1.5
    assert cashflow_features(frame, windows=(30,))["30d_revenue_growth"] == 0.5


def test_missing_day_requires_explicit_choice(daily_path):
    frame = daily_path(days=10).drop(index=3)
    with pytest.raises(ValueError, match="missing calendar"):
        cashflow_features(frame, windows=(10,))
    values = cashflow_features(frame, windows=(10,), missing_days="zero")
    assert values["10d_revenue_sum"] == 900
    assert values["10d_zero_revenue_share"] == 0.1
    assert values["10d_operating_cash_sum"] == 360


def test_panel_isolated_per_merchant(daily_path):
    a = daily_path(days=30, revenue=100).assign(merchant_id="a")
    b = daily_path(days=30, revenue=200).assign(merchant_id="b")
    result = merchant_features(pd.concat([a, b]), windows=(30,), as_of="2025-01-30")
    assert result.loc["b", "30d_revenue_sum"] == 2 * result.loc["a", "30d_revenue_sum"]


def test_inputs_are_not_modified_and_unsorted_rows_are_supported(daily_path):
    frame = daily_path(days=30).iloc[::-1]
    original = frame.copy(deep=True)
    cashflow_features(frame)
    pd.testing.assert_frame_equal(frame, original)


@pytest.mark.parametrize("value", [-1, np.nan, np.inf])
@pytest.mark.parametrize("column", ["revenue", "operating_cost"])
def test_invalid_monetary_inputs(daily_path, value, column):
    frame = daily_path()
    frame.loc[0, column] = value
    with pytest.raises(ValueError, match=column):
        cashflow_features(frame)


@pytest.mark.parametrize("windows", [(), (30, 30), (0,), (True,), (2.5,)])
def test_invalid_windows(daily_path, windows):
    with pytest.raises(ValueError):
        cashflow_features(daily_path(), windows=windows)


def test_duplicate_dates_are_rejected(daily_path):
    frame = daily_path()
    with pytest.raises(ValueError, match="one row"):
        cashflow_features(pd.concat([frame, frame.iloc[:1]]))


@pytest.mark.parametrize("stamp", ["2025-01-01T12:00", "2025-01-01T00:00Z", None])
def test_calendar_date_contract(stamp):
    with pytest.raises(ValueError):
        cashflow_features(pd.DataFrame({"date": [stamp], "revenue": [100]}))


def test_missing_merchant_id_rejected(daily_path):
    with pytest.raises(ValueError, match="identifiers"):
        merchant_features(daily_path().assign(merchant_id=None))
