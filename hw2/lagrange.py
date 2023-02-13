import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

np.random.seed(42069)
x = np.random.uniform(0, 20, 17)
y = np.sin(x)
poly = lagrange(x, y)

yhat_train = Polynomial(poly.coef[::-1])(x)

train_error = y - yhat_train
train_error
mse_train = np.mean(train_error)

x_new = np.arange(0, 20.01, .01)
plt.scatter(x, y, label="data")
plt.plot(x_new, Polynomial(poly.coef[::-1])(x_new), label="Polynomial")
plt.legend()
plt.show()

def experiment(noise):
    
    ini
    
    test_x = np.random.uniform(0, 20, 17)
    noise_x = test_x + 




