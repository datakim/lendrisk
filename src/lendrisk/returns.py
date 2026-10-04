"""Dated, conventional cash-flow returns using an ACT/365 fixed convention."""

import numpy as np
import pandas as pd

from ._validation import day, number, vector


def _flows(amounts: object, dates: object) -> tuple[np.ndarray, np.ndarray]:
    values = vector(amounts, "amounts")
    try:
        stamps = pd.DatetimeIndex([day(d) for d in dates])
    except TypeError as exc:
        raise ValueError("dates must be a sequence of calendar dates") from exc
    if len(stamps) != len(values):
        raise ValueError("amounts and dates must have equal length")
    grouped = pd.Series(values, index=stamps).groupby(level=0).sum().sort_index()
    if not np.isfinite(grouped.to_numpy()).all():
        raise ValueError("net cash flows exceed floating-point range")
    years = (grouped.index - grouped.index[0]).days.to_numpy(dtype=float) / 365.0
    return grouped.to_numpy(dtype=float), years


def xnpv(rate: float, amounts: object, dates: object) -> float:
    """Net present value at an effective annual rate greater than -1."""
    rate = number(rate, "rate", minimum=-1, strict=True)
    values, years = _flows(amounts, dates)
    with np.errstate(over="ignore", invalid="ignore"):
        result = float(np.sum(values * np.exp(-np.log1p(rate) * years)))
    if not np.isfinite(result):
        raise ValueError("discounted cash flows exceed floating-point range")
    return result


def xirr(amounts: object, dates: object) -> float:
    """Effective annual IRR for one initial outflow followed by nonnegative inflows.

    Equal-day flows are netted before checking the sign pattern. Nonconventional
    flows are rejected because they can have multiple IRRs. Uses ACT/365 fixed;
    this mathematical return is not a jurisdiction-specific statutory APR.
    """
    values, years = _flows(amounts, dates)
    nonzero = values != 0
    values, years = values[nonzero], years[nonzero]
    if not len(values):
        raise ValueError("net cash flows cannot all be zero")
    years = years - years[0]
    if len(values) < 2 or values[0] >= 0 or (values[1:] < 0).any() or not (years[1:] > 0).any():
        raise ValueError("xirr requires one initial outflow followed by later nonnegative inflows")

    def objective(log_rate: float) -> float:
        with np.errstate(over="ignore"):
            return float(np.sum(values * np.exp(-log_rate * years)))

    lower, upper = -1.0, 1.0
    while objective(lower) < 0 and lower > -512:
        lower *= 2
    while objective(upper) > 0 and upper < 512:
        upper *= 2
    if objective(lower) < 0 or objective(upper) > 0:
        raise ValueError("could not bracket xirr within floating-point range")
    for _ in range(200):
        middle = (lower + upper) / 2
        if objective(middle) > 0:
            lower = middle
        else:
            upper = middle
        if upper - lower < 1e-13:
            break
    result = float(np.expm1((lower + upper) / 2))
    if not np.isfinite(result) or result <= -1:
        raise ValueError("xirr exceeds floating-point range")
    return result
