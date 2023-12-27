import base64
import math
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import copy
from Model import Model


def compute_gradient(features, targets, weight, bias, _lambda=1) -> [float, float]:
    num_of_training_data, num_of_training_features = features.shape
    num_of_features = len(weight)

    dj_dw = np.zeros((num_of_training_features,))
    dj_db = 0.
    for i in range(num_of_training_data):
        error = (np.dot(features[i], weight) + bias) - targets[i]
        for j in range(num_of_training_features):
            dj_dw[j] = dj_dw[j] + (error * features[i, j])
        dj_db = dj_db + error
    dj_dw = dj_dw / num_of_training_data
    dj_db = dj_db / num_of_training_data
    for j in range(num_of_features):
        dj_dw[j] = dj_dw[j] + (_lambda / num_of_training_data) * weight[j]

    return dj_dw, dj_db


def compute_cost(features, targets, weight, bias, _lambda=1) -> float:
    num_of_training_data = features.shape[0]
    num_of_features = len(weight)
    cost = 0.0
    for i in range(num_of_training_data):
        f_wb_i = np.dot(features[i], weight) + bias  # XW + B
        cost = cost + (f_wb_i - targets[i]) ** 2
    cost = cost / (2 * num_of_training_data)
    reg_cost = 0
    for j in range(num_of_features):
        reg_cost += (weight[j] ** 2)
    reg_cost = (_lambda / (2 * num_of_training_data)) * reg_cost

    total_cost = cost + reg_cost
    return total_cost


def gradient_descent(features, targets, weight_in, bias_in, cost_function, gradient_function, alpha,
                     number_of_iterations, _lambda=1):
    weight = copy.deepcopy(weight_in)
    bias = bias_in
    cost_history = []
    for iteration in range(number_of_iterations):
        dj_dw, dj_db = gradient_function(features, targets, weight, bias, _lambda)
        weight = weight - alpha * dj_dw
        bias = bias - alpha * dj_db
        cost_history.append(cost_function(features, targets, weight, bias, _lambda))
        if iteration % 100 == 0:
            print(f"Iteration {iteration:4d}: Cost {cost_history[-1]}")
    return weight, bias, cost_history


def fit(features, targets, alpha, number_of_iterations,_lambda=1) -> Model:
    num_features = features.shape[1]  # Number of features
    weight = np.zeros(num_features)
    weight, bias, cost_history = gradient_descent(features, targets, weight, 0., compute_cost, compute_gradient,
                                                  alpha, number_of_iterations,_lambda)
    model = Model()
    model.cost_history = cost_history
    model.bias = bias
    model.weight = weight

    return model

def generate_polynomial_features(poly_coeffs, features):
    result = []
    result.extend(features)
    for coeff in poly_coeffs:
        result.extend(np.power(features, coeff))

    return np.array(result)
def predict(model: Model, features):
    if model.polynomials and len(model.polynomials) !=0:
        features = generate_polynomial_features(model.polynomials,features)
    norm_features = zscore_normalize_features_with_parm(features,model.x_mu,model.x_sigma)
    norm_targets = np.dot(norm_features, model.weight) + model.bias
    targets = zscore_unnormalize_features(norm_targets,model.y_mu,model.y_sigma)
    return targets


def zscore_normalize_features(data):
    # find the mean of each column/feature
    mu = np.mean(data, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(data, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    data_norm = (data - mu) / sigma

    return (data_norm, mu, sigma)

def zscore_normalize_features_with_parm(data,mu, sigma):
    data_norm = (data - mu) / sigma

    return data_norm

def zscore_unnormalize_features(data_norm, mu, sigma):
    # element-wise, multiply by std for that column, add mu for that column
    data = data_norm * sigma + mu

    return data


def show_history(model):
    # examin cost history
    fig, ax1  = plt.subplots(1, 1, constrained_layout=True, figsize=(18,5))
    fig.patch.set_facecolor("#0b0f19")
    ax1.set_facecolor("#0b0f19")
    ax1.plot(model.cost_history, color="white")
    ax1.set_title("Cost vs. iteration = {}".format(round(model.cost_history[-1], 4)),color="white")
    ax1.set_ylabel('Cost',color="white")
    ax1.set_xlabel('iteration step',color="white")
    ax1.tick_params(colors="white")
    ax1.spines["top"].set_color("#0b0f19")
    ax1.spines["right"].set_color("#0b0f19")
    ax1.spines["bottom"].set_color("white")
    ax1.spines["left"].set_color("white")
    # Convert the plot to an image
    plt.savefig("image.png", format='png')

    return "image.png"
