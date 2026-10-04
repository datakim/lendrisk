"""Dated, conventional cash-flow returns using an ACT/365 fixed convention."""

import numpy as np
import pandas as pd

from ._validation import day, number, vector


class UnrepresentableReturnError(ValueError):
    """The conventional IRR is outside the finite float range above -1."""


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
    log_discount = -np.log1p(rate) * years
    extreme = (log_discount > np.log(np.finfo(float).max)) | (
        log_discount < np.log(np.finfo(float).tiny)
    )
    weighted = np.zeros_like(values)
    ordinary = ~extreme & (values != 0)
    exceptional = extreme & (values != 0)
    with np.errstate(over="ignore", invalid="ignore"):
        weighted[ordinary] = values[ordinary] * np.exp(log_discount[ordinary])
        weighted[exceptional] = np.sign(values[exceptional]) * np.exp(
            np.log(np.abs(values[exceptional])) + log_discount[exceptional]
        )
        result = float(np.sum(weighted))
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

    log_inflows = np.log(values[1:])
    log_outflow = np.log(-values[0])

    def objective(log_rate: float) -> float:
        # Compare log present values instead of exponentiating discount factors.
        # This retains tiny/large amounts whose intermediate products overflow
        # or underflow even though the final annual return is representable.
        return float(np.logaddexp.reduce(log_inflows - log_rate * years[1:]) - log_outflow)

    lower, upper = -1.0, 1.0
    minimum = float(np.log1p(np.nextafter(-1.0, 0.0)))
    maximum = float(np.log(np.finfo(float).max))
    while objective(lower) < 0 and lower > minimum:
        lower = max(2 * lower, minimum)
    while objective(upper) > 0 and upper < maximum:
        upper = min(2 * upper, maximum)
    if objective(lower) < 0 or objective(upper) > 0:
        raise UnrepresentableReturnError("xirr is outside the finite floating-point range")
    for _ in range(200):
        middle = (lower + upper) / 2
        if objective(middle) > 0:
            lower = middle
        else:
            upper = middle
        if upper - lower < 1e-13:
            break
    with np.errstate(over="ignore"):
        result = float(np.expm1((lower + upper) / 2))
    if not np.isfinite(result) or result <= -1:
        raise UnrepresentableReturnError("xirr is outside the finite floating-point range")
    return result
