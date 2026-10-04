"""Native WoE/logistic scorecards with points-to-double-the-odds scaling."""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.linear_model import LogisticRegression
from sklearn.utils.validation import check_is_fitted

from ._validation import integer, number
from .binning import BinningProcess


class LogisticScorecard(ClassifierMixin, BaseEstimator):
    """Fit native bins and logistic regression within a training boundary.

    Higher points mean lower default risk. base_odds denotes good:bad odds at
    base_score; pdo points double those odds. Inputs are numerical DataFrames.
    """

    def __init__(
        self, *, binning_process=None, pdo=20, base_score=600, base_odds=50, C=1.0, max_iter=1000
    ):
        self.binning_process = binning_process
        self.pdo = pdo
        self.base_score = base_score
        self.base_odds = base_odds
        self.C = C
        self.max_iter = max_iter

    def fit(self, X, y):
        for attribute in (
            "estimator_",
            "binning_process_",
            "classes_",
            "feature_names_in_",
            "n_features_in_",
        ):
            self.__dict__.pop(attribute, None)
        number(self.pdo, "pdo", strict=True)
        number(self.base_score, "base_score", minimum=-np.inf)
        number(self.base_odds, "base_odds", strict=True)
        number(self.C, "C", strict=True)
        integer(self.max_iter, "max_iter")
        prototype = self.binning_process if self.binning_process is not None else BinningProcess()
        if not isinstance(prototype, BinningProcess):
            raise ValueError("binning_process must be a BinningProcess")
        process = clone(prototype).fit(X, y)
        woe = process.transform(X)
        estimator = LogisticRegression(C=self.C, max_iter=self.max_iter).fit(woe, y)
        self.binning_process_, self.estimator_ = process, estimator
        self.classes_ = self.estimator_.classes_
        self.feature_names_in_ = self.binning_process_.feature_names_in_
        self.n_features_in_ = len(self.feature_names_in_)
        return self

    def __sklearn_is_fitted__(self):
        return hasattr(self, "estimator_")

    def _more_tags(self):
        return {"allow_nan": True, "binary_only": True}

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        tags.input_tags.allow_nan = True
        tags.classifier_tags.multi_class = False
        return tags

    def predict_proba(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict_proba(self.binning_process_.transform(X))

    def predict(self, X):
        check_is_fitted(self, "estimator_")
        return self.estimator_.predict(self.binning_process_.transform(X))

    def score_points(self, X):
        """Return unrounded points from log odds, avoiding PD clipping."""
        check_is_fitted(self, "estimator_")
        log_bad_odds = self.estimator_.decision_function(self.binning_process_.transform(X))
        return self.base_score + self.pdo / np.log(2) * (-log_bad_odds - np.log(self.base_odds))

    @property
    def intercept_points(self):
        check_is_fitted(self, "estimator_")
        factor = self.pdo / np.log(2)
        return float(
            self.base_score - factor * (self.estimator_.intercept_[0] + np.log(self.base_odds))
        )

    def table(self):
        """Per-bin point contributions; add intercept_points once per application."""
        check_is_fitted(self, "estimator_")
        tables = []
        factor = self.pdo / np.log(2)
        for name, coefficient in zip(self.feature_names_in_, self.estimator_.coef_[0], strict=True):
            table = self.binning_process_.binning_models_[name].table()
            table.insert(0, "variable", name)
            table["points"] = -factor * coefficient * table["woe"]
            tables.append(table)
        return pd.concat(tables, ignore_index=True)
