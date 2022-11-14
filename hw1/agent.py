import typing

import joblib
from sklearn.neural_network import MLPClassifier, MLPRegressor
from sklearn.preprocessing import LabelBinarizer


class Agent:
    def __init__(
            self,
            continuous: bool = False,
            **model_kwargs
    ):
        if not continuous:
            self.model = MLPClassifier(**model_kwargs)
        else:
            self.model = MLPRegressor(**model_kwargs)

    def predict_proba(self, X):
        assert isinstance(self.model, MLPClassifier), f'{self.model.__class__.__name__} cannot predict probabilities'

        return self.model.predict_proba(X)

    def partial_fit(self, X, y, num_classes=None):
        return self.model.partial_fit(X, y, num_classes)

    def fit(self, X, y):
        return self.model.fit(X, y)

    @staticmethod
    def load(path) -> "Agent":
        return joblib.load(path)

    def save(self, path):
        joblib.dump(self, path)
