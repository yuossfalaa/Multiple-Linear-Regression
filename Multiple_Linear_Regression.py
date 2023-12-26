import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from Model import Model


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
    cost = 0.0
    for i in range(num_of_training_data):
        f_wb_i = np.dot(features[i], weight) + bias  # XW + B
        cost = cost + (f_wb_i - targets[i]) ** 2
    cost = cost / (2 * num_of_training_data)
    return cost


def gradient_descent(features, targets, weight_in, bias_in, cost_function, gradient_function, alpha,
                     number_of_iterations):
    weight = copy.deepcopy(weight_in)
    bias = bias_in
    cost_history = []
    for iteration in range(number_of_iterations):
        dj_dw, dj_db = gradient_function(features, targets, weight, bias)
        weight = weight - alpha * dj_dw
        bias = bias - alpha * dj_db
        if iteration < 10000:
            cost_history.append(cost_function(features, targets, weight, bias))
        if iteration % math.ceil(number_of_iterations / 10):
            print(f"Iteration {iteration:4d}: Cost {cost_history[-1]}")
    return weight, bias, cost_history


def fit(features, targets, alpha, number_of_iterations) -> Model:
    num_features = features.shape[1]  # Number of features
    weight = np.zeros(num_features)
    weight, bias, cost_history = gradient_descent(features, targets, weight, 0., compute_cost, compute_gradient,
                                                  alpha, number_of_iterations)
    model = Model()
    model.cost_history = cost_history
    model.bias = bias
    model.weight = weight

    return model


def predict(model: Model, features):
    targets = np.dot(features, model.weight) + model.bias
    return targets


def zscore_normalize_features(data):
    # find the mean of each column/feature
    mu = np.mean(data, axis=0)  # mu will have shape (n,)
    # find the standard deviation of each column/feature
    sigma = np.std(data, axis=0)  # sigma will have shape (n,)
    # element-wise, subtract mu for that column from each example, divide by std for that column
    data_norm = (data - mu) / sigma

    return (data_norm, mu, sigma)


def zscore_unnormalize_features(data_norm, mu, sigma):
    # element-wise, multiply by std for that column, add mu for that column
    data = data_norm * sigma + mu

    return data


if __name__ == "__main__":
    # prepare data
    data = np.genfromtxt('Real estate.csv', delimiter=',', skip_header=1)
    #delete unwanted Data
    data = np.delete(data, [0, 1, 5, 6], 1)
    # Normalize data
    data, mu, sigma = zscore_normalize_features(data)
    polynomial_features = np.c_[data[:, :-1], data[:, :-1] ** 2, data[:, :-1] ** 3]
    data = np.c_[polynomial_features, data[:, -1]]
    training_data, testing_data = np.split(data, [int(0.8 * len(data))])
    # init parameters
    alpha = 0.0003
    number_of_iterations = 15_000
    # fit model
    model = fit(training_data[:, :-1], training_data[:, -1], alpha, number_of_iterations)
    targets = testing_data[:, -1]
    # test model
    predicted_targets = predict(model, testing_data[:, :-1])
    for i in range(10):
        print(f"prediction: {predicted_targets[i]:0.2f}, target value: {targets[i]}")
    # examin cost history
    fig, (ax1, ax2) = plt.subplots(1, 2, constrained_layout=True, figsize=(12, 4))
    ax1.plot(model.cost_history)
    ax2.plot(100 + np.arange(len(model.cost_history)), model.cost_history)
    ax1.set_title("Cost vs. iteration")
    ax2.set_title("Cost vs. iteration (tail)")
    ax1.set_ylabel('Cost')
    ax2.set_ylabel('Cost')
    ax1.set_xlabel('iteration step')
    ax2.set_xlabel('iteration step')
    plt.show()
