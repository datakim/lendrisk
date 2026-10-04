"""Shared input contracts; validation never silently fills missing observations."""

from numbers import Integral, Real

import numpy as np
import pandas as pd


def number(value: float, name: str, *, minimum: float = 0, strict: bool = False) -> float:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Real):
        raise ValueError(f"{name} must be a finite number")
    value = float(value)
    if not np.isfinite(value) or (value <= minimum if strict else value < minimum):
        symbol = ">" if strict else ">="
        raise ValueError(f"{name} must be finite and {symbol} {minimum}")
    return value


def integer(value: int, name: str, *, minimum: int = 1) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, Integral) or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")
    return int(value)


def fraction(value: float, name: str, *, positive: bool = False) -> float:
    value = number(value, name, strict=positive)
    if value > 1:
        raise ValueError(f"{name} must be <= 1")
    return value


def day(value: object, name: str = "date") -> pd.Timestamp:
    try:
        stamp = pd.Timestamp(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f"{name} must be a calendar date") from exc
    if pd.isna(stamp) or stamp.tzinfo is not None or stamp != stamp.normalize():
        raise ValueError(f"{name} must be a timezone-naive calendar date at midnight")
    return stamp


def vector(values: object, name: str) -> np.ndarray:
    try:
        result = np.asarray(values, dtype=float)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if result.ndim != 1 or result.size == 0 or not np.isfinite(result).all():
        raise ValueError(f"{name} must be a nonempty, finite, one-dimensional array")
    return result


def daily_frame(
    data: pd.DataFrame, *, as_of: object = None, missing_days: str = "raise"
) -> pd.DataFrame:
    if not isinstance(data, pd.DataFrame) or not {"date", "revenue"}.issubset(data.columns):
        raise ValueError("data must be a DataFrame with date and revenue columns")
    if not data.columns.is_unique:
        raise ValueError("data columns must be unique")
    if missing_days not in {"raise", "zero"}:
        raise ValueError("missing_days must be 'raise' or 'zero'")
    columns = ["date", "revenue"] + (["operating_cost"] if "operating_cost" in data else [])
    frame = data.loc[:, columns].copy()
    try:
        frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    except (ValueError, TypeError) as exc:
        raise ValueError("date must contain valid calendar dates") from exc
    if not pd.api.types.is_datetime64_any_dtype(frame["date"]):
        raise ValueError("date must contain timezone-naive calendar dates")
    if frame["date"].dt.tz is not None:
        raise ValueError("date must be timezone-naive")
    if frame["date"].isna().any() or not frame["date"].eq(frame["date"].dt.normalize()).all():
        raise ValueError("date must contain calendar dates at midnight without missing values")
    if as_of is not None:
        frame = frame.loc[frame["date"] <= day(as_of, "as_of")]
    if frame.empty:
        raise ValueError("data has no observations on or before as_of")
    if frame["date"].duplicated().any():
        raise ValueError("one row per calendar day is required; aggregate transactions first")
    frame = frame.sort_values("date").reset_index(drop=True)
    for column in columns[1:]:
        try:
            values = pd.to_numeric(frame[column], errors="raise").to_numpy(dtype=float)
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{column} must be numeric") from exc
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{column} must be finite and nonnegative")
        frame[column] = values
    dates = pd.date_range(frame["date"].iloc[0], frame["date"].iloc[-1], freq="D")
    if len(dates) != len(frame):
        if missing_days == "raise":
            raise ValueError("missing calendar days; supply explicit zeros or missing_days='zero'")
        frame = frame.set_index("date").reindex(dates, fill_value=0.0)
        frame.index.name = "date"
        frame = frame.reset_index()
    return frame
