import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import lagrange
from numpy.polynomial.polynomial import Polynomial

def experiment(lb=-6, ub=6, n=10, noise_sd=0):
    
    #Training set
    true_x = np.random.uniform(lb, ub, n)
    noise = noise_sd*np.random.normal(size=n)
    train_x = true_x + noise
    train_y = np.sin(true_x)

    #Training
    poly = lagrange(train_x, train_y)
    poly = Polynomial(poly.coef[::-1])
    
    #Plotting model
    # x_new = np.arange(train_x.min(), train_x.max(), (train_x.max()-train_x.min())/100)
    # plt.scatter(train_x, train_y, label="Data")
    # plt.plot(x_new, np.sin(x_new), label = "Sin")
    # plt.plot(x_new, poly(x_new), label="Polynomial")
    # plt.legend()
    # plt.show()
    
    #Training set predictions
    yhat_train = poly(train_x)

    #Training set error
    train_errors = yhat_train - train_y
    rmse_train = np.sqrt(np.mean(train_errors ** 2)) 
    
    #Test set
    test_x = np.random.uniform(0, 20, n)
    test_y = np.sin(test_x)

    #Test set prediction
    yhat_test = poly(test_x)

    #Test error
    test_errors = yhat_test - test_y 
    rmse_test = np.sqrt(np.mean(test_errors ** 2))
    return(rmse_train, rmse_test)


np.random.seed(42069)
train_error = np.empty(100)
test_error = np.empty(100)

for (i, sd) in enumerate(np.arange(0,20,.2)):
    
    train_error[i], test_error[i] = experiment(noise_sd = sd)

train_error_bign = np.empty(100)
test_error_bign = np.empty(100)

for (i, sd) in enumerate(np.arange(0,20,.2)):
    
    train_error_bign[i], test_error_bign[i] = experiment(n=100, noise_sd = sd)



plt.plot(np.arange(0,20,.2), np.log(train_error))
plt.xlabel("Standard deviation of noise in x")
plt.ylabel("log(rmse)")
plt.title("Training error of Lagrange interpolation of 10 data points")
plt.savefig("figs/train_error_10.png")
plt.clf()

plt.plot(np.arange(0,20,.2), np.log(test_error))
plt.xlabel("Standard deviation of noise in x")
plt.ylabel("log(rmse)")
plt.title("Test error of Lagrange interpolation of 10 data points")
plt.savefig("figs/test_error_10.png")
plt.clf()

plt.plot(np.arange(0,20,.2), np.log(train_error_bign))
plt.xlabel("Standard deviation of noise in x")
plt.ylabel("log(rmse)")
plt.title("Training error of Lagrange interpolation of 100 data points")
plt.savefig("figs/train_error_100.png")
plt.clf()

plt.plot(np.arange(0,20,.2), np.log(test_error_bign))
plt.xlabel("Standard deviation of noise in x")
plt.ylabel("log(rmse)")
plt.title("Test error of Lagrange interpolation of 100 data points")
plt.savefig("figs/test_error_100.png")
plt.clf()
