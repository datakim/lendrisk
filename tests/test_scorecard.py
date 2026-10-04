import numpy as np
import pandas as pd
import pytest
from sklearn.base import clone
from sklearn.datasets import make_classification
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.utils.validation import check_is_fitted

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


def test_process_failed_refit_rejects_previous_transformation():
    X, y = training_data()
    process = BinningProcess().fit(X, y)
    with pytest.raises(ValueError):
        process.fit(pd.DataFrame(), y)
    with pytest.raises(NotFittedError):
        check_is_fitted(process)
    with pytest.raises(NotFittedError):
        process.transform(X)
    with pytest.raises(NotFittedError):
        process.get_feature_names_out()
    assert not hasattr(process, "feature_names_in_")
    # A corrected refit can recover normally.
    assert process.fit(X, y).transform(X).shape == X.shape


def test_scorecard_failed_parameter_refit_rejects_stale_predictions():
    X, y = training_data()
    model = LogisticScorecard().fit(X, y).set_params(C=0)
    with pytest.raises(ValueError):
        model.fit(X, y)
    with pytest.raises(NotFittedError):
        check_is_fitted(model)
    for method in (model.predict_proba, model.predict, model.score_points):
        with pytest.raises(NotFittedError):
            method(X)
    with pytest.raises(NotFittedError):
        model.table()
    assert not hasattr(model, "classes_")
    assert not hasattr(model, "binning_process_")
    assert model.set_params(C=1).fit(X, y).predict_proba(X).shape == (len(X), 2)


def test_estimator_tags_match_supported_inputs_and_targets():
    from sklearn import utils

    for estimator in (OptimalBinning(), BinningProcess(), LogisticScorecard()):
        if hasattr(utils, "get_tags"):
            tags = utils.get_tags(estimator)
            assert tags.input_tags.allow_nan
            assert tags.target_tags.required
            if isinstance(estimator, LogisticScorecard):
                assert not tags.classifier_tags.multi_class
            else:
                assert tags.transformer_tags is not None
        else:
            tags = estimator._get_tags()
            assert tags["allow_nan"]
            assert tags["requires_y"]
            if isinstance(estimator, LogisticScorecard):
                assert tags["binary_only"]


def test_pandas_output_in_column_transformer_retains_names_and_index():
    from sklearn.compose import ColumnTransformer

    X, y = training_data()
    X.index = pd.Index(range(1000, 1000 + len(X)), name="application_id")
    X.loc[X.index[:5], "a"] = np.nan
    transform = ColumnTransformer([("risk", BinningProcess(), list(X.columns))]).set_output(
        transform="pandas"
    )
    output = transform.fit_transform(X, y)
    assert output.index.equals(X.index)
    assert output.columns.tolist() == ["risk__a", "risk__b", "risk__c"]
    assert np.isfinite(output.to_numpy()).all()
