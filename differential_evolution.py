# -*- coding: utf-8 -*-

import numpy as np
import simple_autoencoder as ae
from logging import getLogger
from concurrent import futures

logger = getLogger(__name__)


class DE(object):

    def __init__(self, objective_function: callable, ndim: int,
                 lower_limit: np.ndarray, upper_limit: np.ndarray,
                 minimize: bool = True):
        self._of = objective_function
        self._pop = None
        self._nd = ndim
        self._x_current = None
        self._low_lim = lower_limit
        self._up_lim = upper_limit
        self._f_current = None
        self._is_minimize = minimize
        self._orbit = None

    def initialization(self, x_init=None):
        if x_init:
            self._x_current = x_init
        else:
            self._x_current = np.random.rand(
                self._pop, self._nd) * (self._up_lim - self._low_lim) + self._low_lim
        self._orbit = []

    def _selection(self, p, u): ∏
    fu = self._evaluate_with_check(u)
    q1 = fu <= self._f_current[p] if self._is_minimize else fu >= self._f_current[p]
    q2 = np.any(u < self._low_lim)
    q3 = np.any(u > self._up_lim)
    q = q1 * ~q2 * ~q3
    f_p1 = fu if q else self._f_current[p]
    x_p1 = u if q else self._x_current[p]
    return p, f_p1, x_p1

    def _mutation(self, current, mutant, num, sf):
        assert num > 0
        if mutant == "best":
            r_best = np.argmin(self._f_current) if self._is_minimize else np.argmax(
                self._f_current)
            r = [r_best]
            r += np.random.choice([n for n in range(self._pop)
                                   if n != r_best], 2 * num, replace=False).tolist()
            v = self._x_current[r[0]] + sf * np.sum(
                [self._x_current[r[m + 1]] - self._x_current[r[m + 2]] for m in range(num)], axis=0)
        else:
            raise ValueError("mutation Error")
        return v

    def _evaluate_with_check(self, x):
        if np.any(x < self._low_lim) or np.any(x > self._up_lim):
            return np.inf if self._is_minimize else - np.inf
        else:
            try:
                f = self._of(x)
            except Exception as ex:
                logger.error(ex)
                f = np.inf if self._is_minimize else - np.inf
            return f

    def _crossover(self, v, x, cross, cr):
        r = np.random.choice(range(self._nd))
        u = np.zeros(self._nd)

        if cross == "bin":
            flg = np.equal(r, np.arange(self._nd)) + \
                np.random.rand(self._nd) < cr
        elif cross == "exp":
            flg = np.array([False for _ in range(self._nd)])
            for _ in range(self._nd):
                flg[r] = True
                r = (r + 1) % self._nd
                if np.random.rand() >= cr:
                    break
        else:
            raise ValueError("")
        u[flg] = v[flg]
        u[~flg] = x[~flg]
        return u

    def _mutation_crossover(self, mutant, num, sf, cross, cr):
        l_up = []
        for p in range(self._pop):
            v_p = self._mutation(p, mutant=mutant, num=num, sf=sf)
            u_p = self._crossover(v_p, self._x_current[p], cross=cross, cr=cr)
            l_up.append(u_p)
        return l_up

    def optimize_mp(self, k_max: int, population: int = 10, mutant: str = "best", num: int = 1, cross: str = "bin", sf: float = 0.7, cr: float = 0.3, proc: [int, None] = None):
        self._pop = population
        self.initialization()

        with futures.ProcessPoolExecutor(proc) as executor:
            results = executor.map(self._evaluate, zip(
                range(self._pop), self._x_current))

            self._f_current = np.array([r[1] for r in sorted(list(results))])

            for k in range(k_max):
                l_up = self._mutation_crossover(mutant, num, sf, cross, cr)
                with futures.ProcessPoolExecutor(proc) as executor:
                    results = executor.map(
                        self._selection, range(self._pop), l_up)

                _x_current = []
                _f_current = []
                for _, fp, x in sorted(results):
                    _x_current.append(x)
                    _f_current.append(fp)

                self._x_current = np.r_[_x_current].copy()
                self._f_current = np.array(_f_current).copy()

                best_score = np.amin(
                    self._f_current) if self._is_minimize else np.amax(self._f_current)
                logger.info("k={} best score={}".format(k, best_score))
                self._orbit.append(best_score)

            best_idx = np.argmin(
                self._f_current) if self._is_minimize else np.argmax(self._f_current)
            x_best = self._x_current[best_idx]
            logger.info("global best score={}".format(
                self._f_current[best_idx]))
            logger.info("x_best = {}".format(x_best))
            return x_best

    def optimize(self, k_max: int, population: int = 10, mutant: str = "best", num: int = 1, cross: str = "bin", sf: float = 0.7, cr: float = 0.3):
        self._pop = population
        self.initialization()
        self._f_current = np.array(
            [self._evaluate_with_check(x) for x in self._x_current])

        for k in range(k_max):
            l_up = self._mutation_crossover(mutant, num, sf, cross, cr)

            for p, u_p in enumerate(l_up):
                _, f_p1, x_p1 = self._selection(p, u_p)
                self._f_current[p] = f_p1
                self._x_current[p] = x_p1

            best_score = np.amin(
                self._f_current) if self._is_minimize else np.amax(self._f_current)
            logger.info("k={} best score={}".format(k, best_score))
            self._orbit.append(best_score)

        best_idx = np.argmin(
            self._f_current) if self._is_minimize else np.amax(self._f_current)
        x_best = self._x_current[best_idx]
        logger.info("global best score = {}".format(self._f_current[best_idx]))
        logger.info("x_best = {}".format(x_best))
        return x_best

    def _evaluate(self, params):
        current, u = params
        return current, self._evaluate_with_check(u)

    @property
    def orbit(self):
        return self.orbit
