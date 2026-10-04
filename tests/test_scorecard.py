import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from lendrisk import BinningProcess, LogisticScorecard, OptimalBinning


def training_data():
    X, y = make_classification(n_samples=240, n_features=3, n_redundant=0, random_state=42)
    return pd.DataFrame(X, columns=["a", "b", "c"]), y


def test_native_scorecard_fit_predict_and_scaling():
    X, y = training_data()
    model = LogisticScorecard().fit(X.iloc[:180], y[:180])
    probabilities = model.predict_proba(X.iloc[180:])[:, 1]
    points = model.score_points(X.iloc[180:])
    assert len(points) == 60
    assert ((probabilities >= 0) & (probabilities <= 1)).all()
    expected = 600 + 20 / np.log(2) * (np.log((1 - probabilities) / probabilities) - np.log(50))
    np.testing.assert_allclose(points, expected)
    assert model.table()["variable"].nunique() == 3
    assert clone(model).get_params() == model.get_params()


def test_point_contributions_reconstruct_total_score():
    X, y = training_data()
    model = LogisticScorecard().fit(X, y)
    contribution = np.full(len(X), model.intercept_points)
    factor = 20 / np.log(2)
    for column, coefficient in zip(X.columns, model.estimator_.coef_[0], strict=True):
        contribution += (
            -factor
            * coefficient
            * model.binning_process_.binning_models_[column].transform(X[column])
        )
    np.testing.assert_allclose(contribution, model.score_points(X))


def test_native_binning_process_in_sklearn_pipeline():
    X, y = training_data()
    pipeline = Pipeline([("binning", BinningProcess()), ("logistic", LogisticRegression())])
    pipeline.fit(X.iloc[:180], y[:180])
    assert pipeline.predict_proba(X.iloc[180:]).shape == (60, 2)
    assert pipeline[:-1].get_feature_names_out().tolist() == list(X.columns)


def test_process_does_not_accept_reordered_columns():
    X, y = training_data()
    process = BinningProcess(binning=OptimalBinning(max_n_prebins=5)).fit(X, y)
    with pytest.raises(ValueError, match="order"):
        process.transform(X[["c", "b", "a"]])
