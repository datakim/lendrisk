"""Credit model diagnostics with event=1 and larger prediction=greater risk."""

from numbers import Integral

import numpy as np
import pandas as pd

from ._validation import integer, number, vector


def credit_metrics(y_true: object, pd_hat: object) -> dict[str, float]:
    """AUC, Gini, tie-aware KS, Brier, and log loss for binary default labels.

    AUC/Gini/KS are NaN for a cohort containing only one class. Brier and log
    loss remain defined. Probabilities are checked, then clipped only for logs.
    No threshold or approval decision is inferred from these diagnostics.
    """
    labels = vector(y_true, "y_true")
    probs = vector(pd_hat, "pd_hat")
    if len(labels) != len(probs) or not np.isin(labels, [0, 1]).all():
        raise ValueError("y_true must be binary 0/1 labels with the same length as pd_hat")
    if ((probs < 0) | (probs > 1)).any():
        raise ValueError("pd_hat must be in [0, 1]")
    positives = float(labels.sum())
    negatives = len(labels) - positives
    auc, ks = np.nan, np.nan
    if positives and negatives:
        ranks = pd.Series(probs).rank(method="average").to_numpy()
        auc = float(
            (ranks[labels == 1].sum() - positives * (positives + 1) / 2) / (positives * negatives)
        )
        grouped = pd.DataFrame({"p": probs, "y": labels}).groupby("p")["y"].agg(["sum", "count"])
        good = (grouped["count"] - grouped["sum"]).cumsum() / negatives
        bad = grouped["sum"].cumsum() / positives
        ks = float((good - bad).abs().max())
    clipped = np.clip(probs, np.finfo(float).eps, 1 - np.finfo(float).eps)
    return {
        "n_observations": float(len(labels)),
        "event_rate": float(labels.mean()),
        "mean_predicted_pd": float(probs.mean()),
        "roc_auc": auc,
        "gini": 2 * auc - 1,
        "ks": ks,
        "brier_score": float(np.mean((probs - labels) ** 2)),
        "log_loss": float(-np.mean(labels * np.log(clipped) + (1 - labels) * np.log1p(-clipped))),
    }


def expected_loss(pd_hat: object, lgd: object, ead: object) -> float | np.ndarray:
    """Return elementwise PD * LGD * EAD with NumPy broadcasting; no discounting.

    PD and LGD must be fractions in [0,1], EAD nonnegative. Sum the returned
    vector yourself when aggregating exposures. Scalars return a float.
    """
    try:
        probabilities, severities, exposures = np.broadcast_arrays(
            np.asarray(pd_hat, dtype=float),
            np.asarray(lgd, dtype=float),
            np.asarray(ead, dtype=float),
        )
    except (ValueError, TypeError) as exc:
        raise ValueError("PD, LGD, and EAD must be numeric and broadcast-compatible") from exc
    for values, name in [(probabilities, "PD"), (severities, "LGD"), (exposures, "EAD")]:
        if not np.isfinite(values).all() or (values < 0).any():
            raise ValueError(f"{name} must be finite and nonnegative")
    if (probabilities > 1).any() or (severities > 1).any():
        raise ValueError("PD and LGD must be fractions in [0, 1]")
    result = probabilities * severities * exposures
    return float(result) if result.ndim == 0 else result


def population_stability_index(
    expected: object,
    actual: object,
    *,
    bins: int | list[float] | np.ndarray = 10,
    pseudocount: float = 0.5,
) -> tuple[float, pd.DataFrame]:
    """PSI with boundaries fitted on expected data and additive count smoothing.

    Integer bins use reference quantiles, collapsing repeated cutpoints. A
    constant reference uses a narrow central bin and two tails. Explicit bins
    are interior cutpoints; +/-infinity are added so shifts outside reference
    ranges stay visible. Returns (psi, bin_table); no universal alert threshold.
    """
    reference, observed = vector(expected, "expected"), vector(actual, "actual")
    pseudocount = number(pseudocount, "pseudocount", strict=True)
    if isinstance(bins, Integral) and not isinstance(bins, (bool, np.bool_)):
        count = integer(bins, "bins", minimum=2)
        if np.all(reference == reference[0]):
            cuts = np.array(
                [np.nextafter(reference[0], -np.inf), np.nextafter(reference[0], np.inf)]
            )
        else:
            cuts = np.unique(np.quantile(reference, np.linspace(0, 1, count + 1)[1:-1]))
    else:
        cuts = vector(bins, "bins")
        if not (np.diff(cuts) > 0).all():
            raise ValueError("bins must be strictly increasing finite interior cutpoints")
    edges = np.r_[-np.inf, cuts, np.inf]
    reference_count = np.histogram(reference, bins=edges)[0]
    actual_count = np.histogram(observed, bins=edges)[0]
    n_bins = len(edges) - 1
    reference_share = (reference_count + pseudocount) / (len(reference) + pseudocount * n_bins)
    actual_share = (actual_count + pseudocount) / (len(observed) + pseudocount * n_bins)
    contribution = (actual_share - reference_share) * np.log(actual_share / reference_share)
    table = pd.DataFrame(
        {
            "lower": edges[:-1],
            "upper": edges[1:],
            "expected_count": reference_count,
            "actual_count": actual_count,
            "expected_share": reference_share,
            "actual_share": actual_share,
            "psi_contribution": contribution,
        }
    )
    return float(contribution.sum()), table
