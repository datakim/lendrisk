import pandas as pd
import pytest

from lendrisk.scorecard import make_scorecard


def test_real_upstream_scorecard_fit_predict():
    pytest.importorskip("optbinning")
    from sklearn.datasets import make_classification

    X, y = make_classification(n_samples=240, n_features=3, n_redundant=0, random_state=42)
    X = pd.DataFrame(X, columns=["a", "b", "c"])
    model = make_scorecard(list(X.columns))
    model.fit(X.iloc[:180], y[:180])
    predictions = model.predict_proba(X.iloc[180:])
    assert predictions.shape == (60, 2)
    assert ((predictions >= 0) & (predictions <= 1)).all()


@pytest.mark.parametrize("names", [[], ["a", "a"], [""], [123]])
def test_bad_variable_names_rejected(names):
    with pytest.raises(ValueError):
        make_scorecard(names)
