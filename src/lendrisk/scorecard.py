"""Optional adapter to OptBinning; its solver remains maintained upstream."""


def make_scorecard(variable_names: list[str], *, estimator: object = None, **kwargs: object):
    """Construct an unfitted upstream scorecard; call fit on training data only.

    Requires pip install 'lendrisk[scorecard]'. kwargs are passed to the upstream
    Scorecard constructor. Bucketing, modeling, and serialization are handled
    by OptBinning; lendrisk does not reproduce the optimal-binning algorithm.
    """
    if not variable_names or len(set(variable_names)) != len(variable_names):
        raise ValueError("variable_names must be nonempty and unique")
    if any(not isinstance(name, str) or not name for name in variable_names):
        raise ValueError("variable_names must contain nonempty strings")
    try:
        from optbinning import BinningProcess, Scorecard
        from sklearn.linear_model import LogisticRegression
    except ImportError as exc:
        raise ImportError(
            "Install the optional dependencies: pip install 'lendrisk[scorecard]'"
        ) from exc
    if estimator is None:
        estimator = LogisticRegression(max_iter=1_000)
    return Scorecard(
        binning_process=BinningProcess(variable_names=variable_names),
        estimator=estimator,
        **kwargs,
    )
