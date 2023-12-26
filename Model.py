import json
import numpy as np
class Model:
    def __int__(self, weight, bias, cost_history):
        self.weight = weight
        self.bias = bias
        self.cost_history = cost_history
        self.x_mu = 0.0
        self.x_sigma = 0.0
        self.y_mu = 0.0
        self.y_sigma = 0.0

    def save(self, json_name):
        data = {
            "weight": self.weight.tolist(),
            "bias": self.bias,
            "cost_history": self.cost_history.tolist(),
            "x_mu": self.x_mu,
            "x_sigma": self.x_sigma,
            "y_mu": self.y_mu,
            "y_sigma": self.y_sigma
        }

        with open(json_name, 'w') as json_file:
            json.dump(data, json_file)

    def load(self, json_name):
        with open(json_name, 'r') as json_file:
            data = json.load(json_file)
        self.weight = np.array(data["weight"])
        self.bias = data["bias"]
        self.cost_history = np.array(data["cost_history"])
        self.x_mu = data["x_mu"]
        self.x_sigma = data["x_sigma"]
        self.y_mu = data["y_mu"]
        self.y_sigma = data["y_sigma"]

