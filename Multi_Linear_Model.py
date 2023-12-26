import math
import numpy as np
import matplotlib.pyplot as plt
import copy
from Model import Model
def prepare_data():
    # prepare data
    data = np.genfromtxt('Real estate.csv', delimiter=',', skip_header=1)
    # delete unwanted Data
    data = np.delete(data, [0, 1, 5, 6], 1)