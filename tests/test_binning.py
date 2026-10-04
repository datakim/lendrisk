from itertools import combinations

import numpy as np
import pytest
from sklearn.exceptions import NotFittedError
from sklearn.utils.validation import check_is_fitted

from lendrisk import BinningOptimizationError, OptimalBinning


def data_from_counts(events, nonevents):
    x, y = [], []
    for i, (bad, good) in enumerate(zip(events, nonevents, strict=True)):
        x.extend([i] * (bad + good))
        y.extend([1] * bad + [0] * good)
    return np.array(x, dtype=float), np.array(y)


def exhaustive_iv(events, nonevents, trend, max_bins=3):
    """Independent enumeration of every possible ordered partition of four groups."""
    n = len(events)
    best = -np.inf
    for k in range(1, max_bins + 1):
        for splits in combinations(range(1, n), k - 1):
            edges = (0, *splits, n)
            bad = np.array([sum(events[a:b]) for a, b in zip(edges, edges[1:], strict=False)])
            good = np.array([sum(nonevents[a:b]) for a, b in zip(edges, edges[1:], strict=False)])
            rates = bad / (bad + good)
            if trend == "ascending" and np.any(np.diff(rates) < 0):
                continue
            if trend == "descending" and np.any(np.diff(rates) > 0):
                continue
            p, q = bad / sum(events), good / sum(nonevents)
            value = np.sum((p - q) * np.log(p / q))
            best = max(best, value)
    return best


@pytest.mark.parametrize("seed", range(5))
@pytest.mark.parametrize("trend", [None, "ascending", "descending", "auto"])
def test_solver_matches_exhaustive_partition_oracle(seed, trend):
    rng = np.random.default_rng(seed)
    events = rng.integers(1, 9, 4).tolist()
    nonevents = rng.integers(1, 9, 4).tolist()
    x, y = data_from_counts(events, nonevents)
    model = OptimalBinning(
        max_n_prebins=4,
        user_splits=[0.5, 1.5, 2.5],
        max_n_bins=3,
        min_bin_size=0.001,
        monotonic_trend=trend,
    ).fit(x, y)
    if trend == "auto":
        expected = max(
            exhaustive_iv(events, nonevents, "ascending"),
            exhaustive_iv(events, nonevents, "descending"),
        )
    else:
        expected = exhaustive_iv(events, nonevents, trend)
    assert model.status_ == "OPTIMAL"
    assert model.objective_gap_ == 0
    assert model.iv_ == pytest.approx(expected, abs=3e-8)


def test_comparison_to_upstream_on_shared_search_space():
    upstream = pytest.importorskip("optbinning")
    x, y = data_from_counts([2, 5, 8, 12], [12, 8, 5, 2])
    common = dict(
        user_splits=[0.5, 1.5, 2.5],
        min_n_bins=1,
        max_n_bins=3,
        min_bin_size=0.05,
        monotonic_trend="ascending",
    )
    ours = OptimalBinning(**common).fit(x, y)
    reference = upstream.OptimalBinning(**common, min_bin_n_event=1, min_bin_n_nonevent=1)
    reference.fit(x, y)
    reference.binning_table.build()
    assert ours.iv_ == pytest.approx(reference.binning_table.iv, abs=1e-6)


def test_missing_special_and_unseen_missing_have_explicit_buckets():
    x, y = data_from_counts([2, 8], [8, 2])
    model = OptimalBinning(max_n_prebins=2, user_splits=[0.5], special_codes=(-999,)).fit(x, y)
    result = model.transform([0, 1, np.nan, -999], metric="indices")
    assert result[-2:].tolist() == [len(model.splits_) + 1, len(model.splits_) + 2]
    assert model.transform([np.nan, -999]).tolist() == [0, 0]
    model.fit(np.r_[x, np.nan, -999], np.r_[y, 1, 0])
    assert model.table()["count"].sum() == len(x) + 2
    assert np.isfinite(model.transform([np.nan, -999])).all()


def test_bin_constraints_and_pure_prebin_merging():
    x, y = data_from_counts([0, 3, 8, 10], [10, 7, 2, 0])
    model = OptimalBinning(
        max_n_prebins=4, user_splits=[0.5, 1.5, 2.5], max_n_bins=3, min_bin_size=0.2
    ).fit(x, y)
    table = model.table().iloc[:-2]
    assert table["event"].ge(1).all()
    assert table["nonevent"].ge(1).all()
    assert table["count_share"].ge(0.2).all()
    assert np.diff(table["event_rate"]).min(initial=0) >= 0


def test_infeasible_fit_raises_without_retaining_old_fitted_state():
    x, y = data_from_counts([2, 8], [8, 2])
    model = OptimalBinning(max_n_prebins=2).fit(x, y)
    model.set_params(min_n_bins=3, max_n_bins=3)
    with pytest.raises(BinningOptimizationError, match="INFEASIBLE"):
        model.fit(x, y)
    assert model.status_ == "INFEASIBLE"
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.transform(x)


def test_invalid_refit_clears_prior_solver_diagnostics_and_prediction():
    x, y = data_from_counts([2, 8], [8, 2])
    model = OptimalBinning().fit(x, y)
    model.set_params(min_n_bins=6, max_n_bins=5)
    with pytest.raises(ValueError):
        model.fit(x, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    with pytest.raises(NotFittedError):
        model.table()
    assert not hasattr(model, "status_")
    assert not hasattr(model, "solver_seconds_")


def test_constant_feature_and_right_open_boundaries():
    model = OptimalBinning().fit([1, 1, 1, 1], [0, 1, 0, 1])
    assert len(model.splits_) == 0
    assert model.iv_ == 0
    x, y = data_from_counts([2, 8], [8, 2])
    model = OptimalBinning(max_n_prebins=2, user_splits=[0.5]).fit(x, y)
    assert model.transform([0.5], metric="indices")[0] == 1


@pytest.mark.parametrize(
    "kwargs",
    [
        {"min_bin_size": 0},
        {"monotonic_trend": "peak"},
        {"max_n_prebins": 51},
        {"user_splits": [1, 1]},
        {"smoothing": 0},
        {"special_codes": [np.nan]},
    ],
)
def test_invalid_parameters(kwargs):
    with pytest.raises(ValueError):
        OptimalBinning(**kwargs).fit([0, 1, 2, 3], [0, 1, 0, 1])
