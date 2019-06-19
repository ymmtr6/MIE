# -*- coding:utf-8 -*-

import numpy as np
from differential_evolution import DE
from tempfile import TemporaryDirectory
import joblib
from pathlib import Path
from logging import getLogger, basicConfig
from sklearn.model_selection import KFold


logger = getLogger(__name__)


class Tuner(object):

    def __init__(self, model, space: dict, k_hold=5, **params):
        self._model = model
        assert isinstance(space, dict)
        self._space = space
        self._parameters = list(self._space.keys())
        self._static_params = [
            p for p in self._parameters if not isinstance(self._space[p], dict)]
        self._variable_params = [
            p for p in self._parameters if isinstance(self._space[p], dict)]
        self._tempdir = TemporaryDirectory()
        self._tempfile = Path(self._tempdir.name + "temp_data.gz")
        self._eval_function = None
        default_opt_param = {
            "k_max": 100,
            "population": 10,
            "mutant": "best",
            "num": 1,
            "cross": "bin",
            "sf": 0.7,
            "cr": 0.4
        }
        self._optimizer_param = default_opt_param
        self._optimizer_param.update(params)
        self._kf = k_hold

    def __del__(self):
        self._tempdir.cleanup()

    def _get_search_limits(self):
        lowers = []
        uppers = []
        for k in self._variable_params:
            if self._space[k]["scale"] in ["linear", "log"]:
                lowers.append(self._space[k]["range"][0])
            elif self._space[k]["scale"] == "integer":
                lowers.append(self._space[k]["range"][0])
                uppers.append(self._space[k]["range"][1])
            else:
                lowers.append(0)
                uppers.append(len(self._space[k]["range"]))

        return np.array(lowers), np.array(uppers)

    def _translate_to_origin(self, x):
        org_x = {}
        for n, k in enumerate(self._variable_params):
            if self._space[k]['scale'] == 'log':
                org_x[k] = np.power(10, x[n])
            elif self._space[k]['scale'] == 'category':
                org_x[k] = self._space[k]['range'][int(x[n])]
            elif self._space[k]['scale'] == 'integer':
                org_x[k] = int(x[n])
            else:
                org_x[k] = x[n]

        # static parameters
        for k in self._static_params:
            org_x[k] = self._space[k]
        return org_x

    def _evaluate(self, x):
        # load data from temporary directory
        input_data, targets = joblib.load(self._tempfile)

        # set model using parameter x
        param = self._translate_to_origin(x)
        model = self._model.set_params(**param)

        # train model using CV (K-fold)
        skf = KFold(n_splits=self._kf, shuffle=True)
        scores = []
        for train, test in skf.split(input_data, targets):
            x_tr, t_tr = input_data[train], targets[train]
            x_te, t_te = input_data[test], targets[test]

            model.fit(x_tr, t_tr)
            scores.append(self._eval_function(
                y_pred=model.predict(x_te), y_true=t_te))

        # average score
        return np.average(scores)

    def tuning(self, eval_function: callable, x: np.ndarray, t: np.ndarray, minimize: bool = True):
        joblib.dump((x, t), self._tempfile)

        # set DE
        lower_limit, upper_limit = self._get_search_limits()

        # set evaluation function
        self._eval_function = eval_function
        optimizer = DE(objective_function=self._evaluate, ndim=len(lower_limit), lower_limit=lower_limit,
                       upper_limit=upper_limit, minimize=minimize)

        x_best = optimizer.optimize_mp(**self._optimizer_param)

        return self._translate_to_origin(x_best)


if __name__ == '__main__':
    basicConfig(level='INFO')

    from sklearn.datasets import load_digits
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import accuracy_score
    from tensorflow.examples.tutorials.mnist import input_data

    search_space = {
        "H": {"scale": "integer", "range": [10, 1000]},
        "BATCH_SIZE": {"scale": "integer", "range": [10, 2000]},
        "DROP_OUT_RATE": {"scale": "linear", "range": [0.01, 0.99]},
        "LEARNING_RATE": {"scale": "log", "range": [1, 9]},
        "BETA1": {"scale": "linear", "range": [0.5, 0.9]},
        "BETA2": {"scale": "linear", "range": [0.5, 0.99]},
        "STDDEV": {"scale": "linear", "range": [0.0, 0.5]},
        "BIAS": {"scale": "linear", "range": [0.0, 0.5]}
    }


"""
    tuner = Tuner(model=RandomForestClassifier(), space=search_space)
    best_param = tuner.tuning(
        eval_function=accuracy_score, x=dataset.data, t=dataset.target, minimize=False)
    logger.info('best parameter = {}'.format(best_param))
"""
