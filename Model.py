import torch
class Model:
    def __int__(self, weight, bias, cost_history):
        self.weight = weight
        self.bias = bias
        self.cost_history = cost_history
        self.x_mu = 0.0
        self.x_sigma = 0.0
        self.y_mu = 0.0
        self.y_sigma = 0.0
    def save (self,json_name):
        pass
