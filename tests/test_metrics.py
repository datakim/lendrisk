import numpy as np
import pytest
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, roc_curve

from lendrisk import credit_metrics, expected_loss, population_stability_index


@pytest.mark.parametrize("seed", range(5))
def test_credit_metrics_against_sklearn_with_ties(seed):
    rng = np.random.default_rng(seed)
    labels = rng.integers(0, 2, 100)
    probs = rng.choice([0.1, 0.2, 0.4, 0.8, 0.9], 100)
    actual = credit_metrics(labels, probs)
    false_positive, true_positive, _ = roc_curve(labels, probs)
    assert actual["roc_auc"] == pytest.approx(roc_auc_score(labels, probs))
    assert actual["ks"] == pytest.approx(np.max(np.abs(true_positive - false_positive)))
    assert actual["brier_score"] == pytest.approx(brier_score_loss(labels, probs))
    assert actual["log_loss"] == pytest.approx(log_loss(labels, probs))


def test_identical_scores_cannot_inflate_ks():
    result = credit_metrics([0, 0, 1, 1], [0.5] * 4)
    assert result["roc_auc"] == 0.5
    assert result["ks"] == 0


def test_single_class_diagnostics_are_explicit():
    result = credit_metrics([0, 0], [0.1, 0.2])
    assert np.isnan(result["roc_auc"])
    assert np.isnan(result["ks"])
    assert result["brier_score"] == pytest.approx(0.025)


@pytest.mark.parametrize(
    "labels,probs",
    [
        ([0, 2], [0.1, 0.2]),
        ([0], [0.1, 0.2]),
        ([0, 1], [-0.1, 0.2]),
        ([0, 1], [np.nan, 0.2]),
        ([], []),
    ],
)
def test_invalid_predictions_rejected(labels, probs):
    with pytest.raises(ValueError):
        credit_metrics(labels, probs)


def test_expected_loss_broadcasting_and_scalar():
    assert expected_loss(0.1, 0.5, 1000) == 50
    np.testing.assert_allclose(expected_loss([0.1, 0.2], 0.5, [1000, 2000]), [50, 200])


@pytest.mark.parametrize(
    "pd_hat,lgd,ead", [(1.1, 0.5, 100), (0.1, -0.5, 100), (0.1, 0.5, -100), (0.1, np.inf, 100)]
)
def test_invalid_expected_loss_inputs(pd_hat, lgd, ead):
    with pytest.raises(ValueError):
        expected_loss(pd_hat, lgd, ead)


def test_psi_zero_for_unchanged_sample():
    psi, table = population_stability_index(np.arange(100), np.arange(100))
    assert psi == 0
    assert table["expected_share"].sum() == pytest.approx(1)


def test_psi_uses_reference_boundaries_and_counts_extreme_shift():
    psi, table = population_stability_index([0, 1, 2, 3], [-100, 100, 100, 100], bins=[1, 2])
    assert table["expected_count"].tolist() == [1, 1, 2]
    assert table["actual_count"].tolist() == [1, 0, 3]
    ref = (np.array([1, 1, 2]) + 0.5) / 5.5
    act = (np.array([1, 0, 3]) + 0.5) / 5.5
    assert psi == pytest.approx(np.sum((act - ref) * np.log(act / ref)))


def test_psi_constant_reference_still_detects_shift():
    assert population_stability_index([1] * 50, [2] * 50)[0] > 0
    assert population_stability_index([1] * 50, [1] * 50)[0] == 0


@pytest.mark.parametrize("bins", [1, [2, 1], [1, 1], [np.inf], True])
def test_invalid_psi_bins(bins):
    with pytest.raises(ValueError):
        population_stability_index([1, 2], [2, 3], bins=bins)
