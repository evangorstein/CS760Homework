import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.neighbors import KNeighborsClassifier

#Read data
data = np.loadtxt("../data/D2z.txt")
X = data[:,:-1]
y = data[:,-1]

#Train classifier (1NN)
neigh = KNeighborsClassifier(n_neighbors = 1)
neigh.fit(X, y)

#Create grid of values and generate predictions over this grid
x_gridvalues, y_gridvalues = np.meshgrid(np.arange(-2,2.1,.1), np.arange(-2,2.1,.1))
test_data = np.vstack([x_gridvalues.flatten(), y_gridvalues.flatten()]).T
y_preds = neigh.predict(test_data)



colors_dict = {0:"blue", 1:"red"}
marker_dict = {0:"o", 1:"x"}
for label in range(2):

    mask_test = y_preds == label
    plt.scatter(test_data[mask_test,0], test_data[mask_test,1], s = .1, color = colors_dict[label])

    mask_train = y == label
    plt.scatter(X[mask_train, 0], X[mask_train, 1], color = "black", 
                marker = marker_dict[label])

plt.show()



#point_colors = [colors_dict[label] for label in y]
#plt.contourf(x_gridvalues, y_gridvalues, y_preds, levels = 1, colors = list(colors_dict.values()))
#plt.scatter(X[:,0], X[:,1], c = point_colors, edgecolors = "black")
#plt.show()

