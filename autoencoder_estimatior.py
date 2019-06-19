from sklearn.base import BaseEstimator


class AEEstimator(BaseEstimator):

    def __init__(self,):
        pass

    def fit(self, x, y):
        return self

    def predict(self, x):
        return [1.0] * len(x)

    def score(self, x, y):
        return 1

    def get_params(self, deep=True):
        pass

    def set_params(self, **parameters):
        for parameter, value in parameters.items():
            setattr(self, parameter, value)
        return self
