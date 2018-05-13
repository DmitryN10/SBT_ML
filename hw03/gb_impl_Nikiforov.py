# coding=utf-8

from sklearn.base import BaseEstimator
from sklearn.tree import DecisionTreeRegressor
from scipy.optimize import minimize, fmin_slsqp
import numpy

# Параметрами с которыми вы хотите обучать деревья
TREE_PARAMS_DICT = {'max_depth': 4}
# Параметр tau (learning_rate) для вашего GB
TAU = 0.05


def loss_function(y_data , curr_pred):
    return sum(numpy.log(1 + numpy.exp(-2 * y_data * curr_pred))) / len(y_data)


def antigrad(y_data, curr_pred):
    return 2 * y_data / (1 + numpy.exp(2 * y_data * curr_pred))


class SimpleGB(BaseEstimator):
    def __init__(self, tree_params_dict, iters, tau):
        self.tree_params_dict = tree_params_dict
        self.iters = iters
        self.tau = tau
        self.estimators = []

    def fit(self, x_data, y_data):
        self.base_algo = DecisionTreeRegressor(**self.tree_params_dict, random_state=1)
        self.base_algo.fit(x_data, y_data)
        curr_pred = self.base_algo.predict(x_data)

        for iter_num in range(self.iters):
            resid = antigrad(y_data, curr_pred)
            algo = DecisionTreeRegressor(**self.tree_params_dict, random_state=1)
            algo.fit(x_data, resid)
            self.estimators.append(algo)
            curr_pred += self.tau * algo.predict(x_data)
        return self

    def predict(self, x_data):
        res = self.base_algo.predict(x_data)
        for estimator in self.estimators:
            res += self.tau * estimator.predict(x_data)
        return res > 1.245