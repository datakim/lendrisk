"""Reproducible synthetic paths: no private merchant data or synthetic credit labels."""

import numpy as np
import pandas as pd

from ._validation import day, fraction, integer, number


def make_merchant_cashflows(
    *,
    days: int = 365,
    start_date: object = "2025-01-01",
    daily_revenue: float = 1_000.0,
    operating_cost_ratio: float = 0.65,
    volatility: float = 0.2,
    seed: int = 42,
) -> pd.DataFrame:
    """Daily revenue with weekly/annual seasonality and mean-one lognormal noise.

    Costs include a fixed component and a variable component. This is an
    illustrative data generator, not a calibrated business or default model.
    All monetary inputs and outputs use the same user-chosen currency unit.
    """
    days = integer(days, "days")
    seed = integer(seed, "seed", minimum=0)
    daily_revenue = number(daily_revenue, "daily_revenue", strict=True)
    cost_ratio = fraction(operating_cost_ratio, "operating_cost_ratio")
    volatility = number(volatility, "volatility")
    if volatility > 5:
        raise ValueError("volatility must be <= 5")
    dates = pd.date_range(day(start_date, "start_date"), periods=days, freq="D")
    rng = np.random.default_rng(seed)
    weekly = np.where(dates.dayofweek >= 5, 1.2, 0.92)
    seasonal = 1 + 0.12 * np.sin(2 * np.pi * (dates.dayofyear - 1) / 365)
    revenue = (
        daily_revenue * weekly * seasonal * rng.lognormal(-(volatility**2) / 2, volatility, days)
    )
    cost = daily_revenue * cost_ratio * 0.6 + revenue * cost_ratio * 0.4
    return pd.DataFrame({"date": dates, "revenue": revenue, "operating_cost": cost})
