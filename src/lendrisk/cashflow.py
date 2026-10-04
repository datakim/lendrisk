"""Point-in-time features from daily merchant revenue and operating costs."""

import numpy as np
import pandas as pd

from ._validation import daily_frame, day, integer


def cashflow_features(
    data: pd.DataFrame,
    *,
    as_of: object = None,
    windows: tuple[int, ...] = (30, 90),
    missing_days: str = "raise",
) -> pd.Series:
    """Summarize one merchant without using observations after ``as_of``.

    Windows include as_of and the preceding window-1 calendar days. Partial
    windows remain partial and expose coverage; unavailable ratios are NaN.
    Revenue CV uses population standard deviation (ddof=0). Growth compares
    total revenue in adjacent windows only when both are complete.
    """
    if not windows or len(set(windows)) != len(windows):
        raise ValueError("windows must be nonempty and unique")
    windows = tuple(integer(w, "window") for w in windows)
    frame = daily_frame(data, as_of=as_of, missing_days=missing_days)
    cutoff = frame["date"].iloc[-1] if as_of is None else day(as_of, "as_of")
    output: dict[str, float] = {}
    for window in windows:
        lower = cutoff - pd.Timedelta(days=window - 1)
        current = frame.loc[frame["date"].between(lower, cutoff)]
        previous = frame.loc[
            frame["date"].between(lower - pd.Timedelta(days=window), lower - pd.Timedelta(days=1))
        ]
        revenue = current["revenue"]
        total = float(revenue.sum()) if len(current) else np.nan
        mean = float(revenue.mean())
        complete = len(current) == window
        prefix = f"{window}d_"
        output[prefix + "observed_days"] = float(len(current))
        output[prefix + "coverage"] = len(current) / window
        output[prefix + "revenue_sum"] = total
        output[prefix + "revenue_mean"] = mean
        output[prefix + "revenue_cv"] = float(revenue.std(ddof=0) / mean) if mean > 0 else np.nan
        output[prefix + "zero_revenue_share"] = float(revenue.eq(0).mean())
        output[prefix + "top_day_share"] = float(revenue.max() / total) if total > 0 else np.nan
        prev_total = float(previous["revenue"].sum())
        output[prefix + "revenue_growth"] = (
            total / prev_total - 1
            if complete and len(previous) == window and prev_total > 0
            else np.nan
        )
        if "operating_cost" in current:
            net = revenue - current["operating_cost"]
            output[prefix + "operating_cash_sum"] = float(net.sum()) if len(current) else np.nan
            output[prefix + "operating_margin"] = float(net.sum() / total) if total > 0 else np.nan
            output[prefix + "negative_operating_cash_share"] = float(net.lt(0).mean())
    return pd.Series(output, dtype=float, name="cashflow_features")


def merchant_features(
    data: pd.DataFrame, *, merchant_column: str = "merchant_id", **kwargs: object
) -> pd.DataFrame:
    """Return one feature row per merchant; as_of should be shared across a cohort."""
    if not isinstance(data, pd.DataFrame) or merchant_column not in data:
        raise ValueError(f"data must contain {merchant_column}")
    if data.empty or data[merchant_column].isna().any():
        raise ValueError("merchant identifiers must be nonmissing in a nonempty panel")
    rows = {
        merchant: cashflow_features(group, **kwargs)
        for merchant, group in data.groupby(merchant_column, sort=False, observed=True)
    }
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = merchant_column
    return result
