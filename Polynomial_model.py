import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from Model import Model
from Multiple_Linear_Regression import zscore_normalize_features, fit




def Train():
    # prepare data
    data = np.genfromtxt('Real estate.csv', delimiter=',', skip_header=1)
    # delete unwanted Data
    data = np.delete(data, [0, 1, 5, 6], 1)
    # Normalize data
    data[:, :-1], x_mu, x_sigma = zscore_normalize_features(data[:, :-1])
    data[:, -1], y_mu, y_sigma = zscore_normalize_features(data[:, -1])
    #adding polynomial features
    polynomial_features = np.c_[data[:, :-1], data[:, :-1] ** 2, data[:, :-1] ** 3]
    data = np.c_[polynomial_features, data[:, -1]]
    print("Data Prepared")
    # split data
    training_data, testing_data = np.split(data, [int(0.8 * len(data))])
    # init parameters
    alpha = 0.0003
    number_of_iterations = 15_000
    # fit model
    model = fit(training_data[:, :-1], training_data[:, -1], alpha, number_of_iterations)
    model.x_mu =x_mu
    model.x_sigma =x_sigma
    model.y_mu =y_mu
    model.y_sigma =y_sigma
    model.save("polynomial_model.json")
    print("Model Saved")
    show_history(model)
def show_history(model):
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
