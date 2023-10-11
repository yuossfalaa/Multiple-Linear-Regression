import math

import numpy as np
import matplotlib as plt
import copy

def compute_gradient(features, targets, weight, bias) -> [float, float]:
    num_of_training_data, num_of_training_features = features.shape
    dj_dw = np.zeros((num_of_training_features,))
    dj_db = 0.
    for i in range(num_of_training_data):
        error = (np.dot(features[i], weight) + bias) - targets[i]
        for j in range(num_of_training_features):
            dj_dw[j] = dj_dw[j] + (error * features[i, j])
        dj_db = dj_db + error
    dj_dw = dj_dw / num_of_training_data
    dj_db = dj_db / num_of_training_data
    return dj_dw, dj_db


def compute_cost(features, targets, weight, bias) -> float:
    num_of_training_data = features.shape[0]
    prediction = np.dot(weight, features) + bias
    cost = (prediction - targets) ** 2
    cost = cost / (2 * num_of_training_data)
    return cost

def gradient_descent(features, targets, weight_in, bias_in, cost_function, gradient_function, alpha, number_of_iterations):
    weight = copy.deepcopy(weight_in)
    bias = bias_in
    cost_history = []
    for iteration in range(number_of_iterations):
        dj_dw, dj_db = gradient_function(features, targets, weight, bias)
        weight = weight - (alpha * dj_dw)
        bias = bias - (alpha * dj_db)
        if iteration < 10000:
            cost_history.append(cost_function(features, targets, weight, bias))
        if iteration % math.ceil(number_of_iterations/10):
            print(f"Iteration {iteration:4d}: Cost {cost_history[-1]:8.2f}")
    return weight, bias , cost_history
