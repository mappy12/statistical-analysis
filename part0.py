import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt

a = -1
sigma2 = 5
sigma = np.sqrt(sigma2)
gamma = 0.90
n = 16
M = 2400
K = 100

np.random.seed(57)
data = np.random.normal(a, sigma, n)

