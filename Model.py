import json
import numpy as np
class Model:
    def __int__(self, weight, bias, cost_history):
        self.weight = weight
        self.bias = bias
        self.cost_history = cost_history
        self.polynomials = []
        self.x_mu = None
        self.x_sigma = None
        self.y_mu = None
        self.y_sigma = None
        self._lambda = 1

    def save(self, json_name):
        data = {
            "weight": self.weight.tolist(),
            "bias": self.bias,
            "cost_history": self.cost_history,
            "polynomials": self.polynomials,
            "x_mu": self.x_mu.tolist(),
            "x_sigma": self.x_sigma.tolist(),
            "y_mu": self.y_mu.tolist(),
            "y_sigma": self.y_sigma.tolist(),
            "_lambda": self._lambda
        }

        with open(json_name, 'w') as json_file:
            json.dump(data, json_file)

    def load(self, json_name):
        with open(json_name, 'r') as json_file:
            data = json.load(json_file)
        self.weight = np.array(data["weight"])
        self.bias = data["bias"]
        self._lambda  = data["_lambda"]
        self.cost_history = data["cost_history"]
        self.polynomials = data["polynomials"]
        self.x_mu = np.array(data["x_mu"])
        self.x_sigma = np.array(data["x_sigma"])
        self.y_mu = np.array(data["y_mu"])
        self.y_sigma = np.array(data["y_sigma"])

    def can_load(self,json_name):
        try:
            with open(json_name, 'r') as json_file:
                return True
        except FileNotFoundError:
            return False