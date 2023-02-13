import sys #Command line arguments
import os #Working with file paths
import numpy as np
import matplotlib.pyplot as plt
from scripts.dec_tree import DecisionTree

def plot_decision_boundary(tree, data, data_name, colors = ["purple", "yellow"]):

    n = len(data)
    colors_dict = {0:colors[0], 1:colors[1]}
    point_colors = [colors_dict[label] for label in data[:,2]]
    
    range_x, range_y = np.ptp(data[:,:2], axis=0)

    x_min, x_max = data[:, 0].min() - range_x/n, data[:, 0].max() + range_x/n
    y_min, y_max = data[:, 1].min() - range_y/n, data[:, 1].max() + range_y/n
    xx, yy = np.meshgrid(np.arange(x_min, x_max, range_x/500),
                         np.arange(y_min, y_max, range_y/500))
    pred = tree.predict(np.c_[xx.ravel(), yy.ravel()])
    pred = pred.reshape(xx.shape)

    plt.contourf(xx, yy, pred, levels = 1, colors = colors)
    plt.scatter(data[:, 0], data[:, 1], c=point_colors, edgecolors="black")
    colorbar = plt.colorbar()
    colorbar.set_label("y")
    plt.xlabel("x0")
    plt.ylabel("x1")
    #plt.xlim(data[:, 0].min(), data[:, 0].max())
    #plt.ylim(data[:, 1].min(), data[:, 1].max())
    plt.savefig("figs/"+data_name+".png")
    plt.show()


if __name__ == "__main__":
    
    data_path = sys.argv[1] #Path to data file given as command argument
    data_name = os.path.splitext(os.path.basename(data_path))[0] 
    data = np.loadtxt(data_path) 
    tree = DecisionTree(data)
    tree.train(data)
    tree.print()
    plot_decision_boundary(tree, data, data_name=data_name)
    

