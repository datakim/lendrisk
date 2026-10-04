"""Deterministic scenario transformations; scenarios are not probability forecasts."""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from ._validation import daily_frame, day, fraction, number
from .products import RevenueAdvance


@dataclass(frozen=True)
class RevenueShock:
    """Multiply revenue in an inclusive calendar interval and preserve fixed costs.

    cost_elasticity specifies the share of costs that scales with revenue.
    Zero keeps all costs fixed; one scales all costs by revenue_multiplier.
    """

    name: str
    revenue_multiplier: float
    start_date: object = None
    end_date: object = None
    cost_elasticity: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip():
            raise ValueError("scenario name must be nonempty")
        number(self.revenue_multiplier, "revenue_multiplier")
        fraction(self.cost_elasticity, "cost_elasticity")
        start = day(self.start_date, "start_date") if self.start_date is not None else None
        end = day(self.end_date, "end_date") if self.end_date is not None else None
        if start is not None and end is not None and start > end:
            raise ValueError("start_date must be on or before end_date")

    def apply(self, data: pd.DataFrame) -> pd.DataFrame:
        """Return a new validated daily path; the input DataFrame is not modified."""
        frame = daily_frame(data)
        mask = pd.Series(True, index=frame.index)
        if self.start_date is not None:
            mask &= frame["date"] >= day(self.start_date)
        if self.end_date is not None:
            mask &= frame["date"] <= day(self.end_date)
        frame.loc[mask, "revenue"] *= self.revenue_multiplier
        if "operating_cost" in frame:
            frame.loc[mask, "operating_cost"] *= (
                1 - self.cost_elasticity + self.cost_elasticity * self.revenue_multiplier
            )
        if not np.isfinite(frame.select_dtypes("number").to_numpy()).all():
            raise ValueError("shocked cash flows exceed floating-point range")
        return frame


def compare_scenarios(
    product: RevenueAdvance,
    data: pd.DataFrame,
    scenarios: list[RevenueShock],
    *,
    opening_cash: float = 0.0,
) -> pd.DataFrame:
    """Compare baseline and named shocks on the same fixed horizon and contract."""
    if not isinstance(product, RevenueAdvance):
        raise ValueError("product must be a RevenueAdvance")
    if any(not isinstance(s, RevenueShock) for s in scenarios):
        raise ValueError("scenarios must contain RevenueShock objects")
    names = [s.name for s in scenarios]
    if len(set(names)) != len(names) or "baseline" in names:
        raise ValueError("scenario names must be unique and cannot be 'baseline'")
    rows = {"baseline": product.simulate(data, opening_cash=opening_cash).summary()}
    for scenario in scenarios:
        rows[scenario.name] = product.simulate(
            scenario.apply(data), opening_cash=opening_cash
        ).summary()
    result = pd.DataFrame.from_dict(rows, orient="index")
    result.index.name = "scenario"
    return result
