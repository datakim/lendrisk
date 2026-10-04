"""Native constrained numerical binning for binary targets, with pandas reports."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.utils.validation import check_is_fitted

from ._binning_solver import candidates_from_counts, solve_partition
from ._validation import fraction, integer, number, vector


class BinningOptimizationError(RuntimeError):
    """No feasible partition was returned under the constraints or time limit."""


def _feature(x):
    try:
        values = pd.to_numeric(pd.Series(x), errors="raise").to_numpy(dtype=float, na_value=np.nan)
    except (TypeError, ValueError) as exc:
        raise ValueError("x must be a one-dimensional numerical feature") from exc
    if not len(values) or np.isinf(values).any():
        raise ValueError("x must be nonempty and cannot contain infinity")
    return values


class OptimalBinning(TransformerMixin, BaseEstimator):
    """Maximize IV over contiguous numerical prebins using the native CP-SAT model.

    Constraints apply to regular observations; Missing/Special are separate
    report buckets. Quantile or supplied boundaries define the search space.
    'auto' solves ascending and descending trends and keeps the better feasible
    objective. Report WoE uses additive smoothing; optimization uses unsmoothed
    regular class counts. This is a focused binary implementation.
    """

    def __init__(
        self,
        *,
        max_n_prebins=20,
        min_n_bins=1,
        max_n_bins=5,
        min_bin_size=0.05,
        max_bin_size=1.0,
        min_bin_n_event=1,
        min_bin_n_nonevent=1,
        monotonic_trend="auto",
        min_event_rate_diff=0.0,
        user_splits=None,
        special_codes=(),
        smoothing=0.5,
        time_limit=30.0,
    ):
        self.max_n_prebins = max_n_prebins
        self.min_n_bins = min_n_bins
        self.max_n_bins = max_n_bins
        self.min_bin_size = min_bin_size
        self.max_bin_size = max_bin_size
        self.min_bin_n_event = min_bin_n_event
        self.min_bin_n_nonevent = min_bin_n_nonevent
        self.monotonic_trend = monotonic_trend
        self.min_event_rate_diff = min_event_rate_diff
        self.user_splits = user_splits
        self.special_codes = special_codes
        self.smoothing = smoothing
        self.time_limit = time_limit

    def fit(self, x, y):
        """Learn boundaries from training observations only; event=1 means default."""
        for attribute in (
            "splits_",
            "prebin_splits_",
            "binning_table_",
            "iv_",
            "objective_gap_",
            "trend_",
            "status_",
            "solver_seconds_",
        ):
            self.__dict__.pop(attribute, None)
        max_prebins = integer(self.max_n_prebins, "max_n_prebins", minimum=2)
        if max_prebins > 50:
            raise ValueError("max_n_prebins must be <= 50 for the interval formulation")
        min_bins = integer(self.min_n_bins, "min_n_bins")
        max_bins = integer(self.max_n_bins, "max_n_bins")
        min_size = fraction(self.min_bin_size, "min_bin_size", positive=True)
        max_size = fraction(self.max_bin_size, "max_bin_size", positive=True)
        if min_bins > max_bins or min_size > max_size:
            raise ValueError("minimum constraints cannot exceed maximum constraints")
        min_event = integer(self.min_bin_n_event, "min_bin_n_event")
        min_nonevent = integer(self.min_bin_n_nonevent, "min_bin_n_nonevent")
        smoothing = number(self.smoothing, "smoothing", strict=True)
        time_limit = number(self.time_limit, "time_limit", strict=True)
        gap = fraction(self.min_event_rate_diff, "min_event_rate_diff")
        if self.monotonic_trend not in (None, "ascending", "descending", "auto"):
            raise ValueError("monotonic_trend must be None, ascending, descending, or auto")
        values, labels = _feature(x), vector(y, "y")
        if len(values) != len(labels) or not np.isin(labels, [0, 1]).all():
            raise ValueError("y must contain binary labels matching x")
        if len(np.unique(labels)) != 2:
            raise ValueError("both event and nonevent labels are required")
        special = np.asarray(self.special_codes, dtype=float)
        if special.ndim != 1 or not np.isfinite(special).all():
            raise ValueError("special_codes must be finite numerical codes")
        missing_mask, special_mask = np.isnan(values), np.isin(values, special)
        regular = ~missing_mask & ~special_mask
        clean, target = values[regular], labels[regular]
        if not len(clean) or len(np.unique(target)) != 2:
            raise ValueError("regular observations must contain both target classes")
        if self.user_splits is not None:
            cuts = vector(self.user_splits, "user_splits")
            if len(cuts) >= max_prebins or not (np.diff(cuts) > 0).all():
                raise ValueError("user_splits must increase strictly and fit max_n_prebins")
        else:
            cuts = np.unique(np.quantile(clean, np.linspace(0, 1, max_prebins + 1)[1:-1]))
            cuts = cuts[(cuts > clean.min()) & (cuts < clean.max())]
        self.prebin_splits_ = cuts.copy()
        prebin_index = np.searchsorted(cuts, clean, side="right")
        n_prebins = len(cuts) + 1
        events = np.bincount(prebin_index, weights=target, minlength=n_prebins).astype(np.int64)
        counts = np.bincount(prebin_index, minlength=n_prebins)
        candidates = candidates_from_counts(
            events,
            counts - events,
            int(np.ceil(min_size * len(clean))),
            int(np.floor(max_size * len(clean))),
            min_event,
            min_nonevent,
        )
        trends = (
            ["ascending", "descending"]
            if self.monotonic_trend == "auto"
            else [self.monotonic_trend]
        )
        results = [
            solve_partition(candidates, n_prebins, min_bins, max_bins, trend, gap, time_limit)
            for trend in trends
        ]
        feasible = [i for i, r in enumerate(results) if r["chosen"]]
        if not feasible:
            self.status_ = (
                "INFEASIBLE" if all(r["status"] == "INFEASIBLE" for r in results) else "UNKNOWN"
            )
            raise BinningOptimizationError(
                f"{self.status_}: no feasible partition; review constraints or time_limit"
            )
        best = max(feasible, key=lambda i: results[i]["objective"])
        result, chosen = results[best], results[best]["chosen"]
        self.status_ = (
            "OPTIMAL"
            if all(r["status"] in ("OPTIMAL", "INFEASIBLE") for r in results)
            else "FEASIBLE"
        )
        self.trend_ = trends[best]
        self.splits_ = np.array([cuts[c.end] for c in chosen[:-1]], dtype=float)
        self.iv_ = float(sum(c.iv for c in chosen))
        bounds = [r["bound"] for r in results if r["status"] != "INFEASIBLE"]
        self.objective_gap_ = max(0.0, float(max(bounds) - result["objective"]))
        self.solver_seconds_ = float(sum(r["seconds"] for r in results))
        k = len(chosen)
        index = np.searchsorted(self.splits_, values, side="right")
        index[missing_mask], index[special_mask] = k, k + 1
        event = np.bincount(index, weights=labels, minlength=k + 2)
        count = np.bincount(index, minlength=k + 2)
        nonevent = count - event
        occupied = count > 0
        n_occupied = int(occupied.sum())
        bad_share = (event + smoothing) / (labels.sum() + smoothing * n_occupied)
        good_share = (nonevent + smoothing) / ((1 - labels).sum() + smoothing * n_occupied)
        woe = np.where(occupied, np.log(good_share / bad_share), 0.0)
        iv = np.where(occupied, (good_share - bad_share) * woe, 0.0)
        rate = np.divide(event, count, out=np.full(k + 2, labels.mean()), where=occupied)
        edges = np.r_[-np.inf, self.splits_, np.inf]
        names = [f"[{edges[i]:.6g}, {edges[i + 1]:.6g})" for i in range(k)] + ["Missing", "Special"]
        self.binning_table_ = pd.DataFrame(
            {
                "bin": names,
                "count": count,
                "count_share": count / len(labels),
                "nonevent": nonevent.astype(int),
                "event": event.astype(int),
                "event_rate": rate,
                "woe": woe,
                "iv": iv,
            }
        )
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "binning_table_")

    def _more_tags(self):
        # scikit-learn 1.4/1.5 use the legacy tag interface.
        return {"requires_y": True, "allow_nan": True, "X_types": ["1darray"]}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = True
        tags.input_tags.one_d_array = True
        tags.input_tags.two_d_array = False
        return tags

    def transform(self, x, *, metric="woe"):
        """Apply learned intervals; NaN/special codes retain dedicated bucket IDs."""
        check_is_fitted(self, "binning_table_")
        if metric not in ("woe", "event_rate", "indices"):
            raise ValueError("metric must be woe, event_rate, or indices")
        values = _feature(x)
        index = np.searchsorted(self.splits_, values, side="right")
        k = len(self.splits_) + 1
        index[np.isnan(values)] = k
        index[np.isin(values, self.special_codes)] = k + 1
        return index if metric == "indices" else self.binning_table_[metric].to_numpy()[index]

    def table(self):
        """Return a copy of the fitted regular/missing/special bin report."""
        check_is_fitted(self, "binning_table_")
        return self.binning_table_.copy(deep=True)


class BinningProcess(TransformerMixin, BaseEstimator):
    """Learn a native OptimalBinning per named numerical DataFrame column."""

    def __init__(self, *, binning=None):
        self.binning = binning

    def fit(self, X, y):
        for attribute in ("binning_models_", "feature_names_in_", "n_features_in_"):
            self.__dict__.pop(attribute, None)
        if not isinstance(X, pd.DataFrame) or X.empty or not X.columns.is_unique:
            raise ValueError("X must be a nonempty DataFrame with unique columns")
        if not all(isinstance(c, str) for c in X.columns):
            raise ValueError("feature names must be strings")
        prototype = self.binning if self.binning is not None else OptimalBinning()
        if not isinstance(prototype, OptimalBinning):
            raise ValueError("binning must be an OptimalBinning estimator")
        models = {name: clone(prototype).fit(X[name], y) for name in X.columns}
        self.feature_names_in_ = X.columns.to_numpy(copy=True)
        self.n_features_in_ = len(X.columns)
        self.binning_models_ = models
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "binning_models_")

    def _more_tags(self):
        return {"requires_y": True, "allow_nan": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.target_tags.required = True
        tags.input_tags.allow_nan = True
        return tags

    def transform(self, X):
        check_is_fitted(self, "binning_models_")
        if not isinstance(X, pd.DataFrame) or list(X.columns) != list(self.feature_names_in_):
            raise ValueError("X columns must match fitted feature names and order")
        return pd.DataFrame(
            {name: model.transform(X[name]) for name, model in self.binning_models_.items()},
            index=X.index,
        )

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self, "binning_models_")
        if input_features is not None and list(input_features) != list(self.feature_names_in_):
            raise ValueError("input_features must match fitted names")
        return self.feature_names_in_.copy()
