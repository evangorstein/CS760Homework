import numpy as np
import matplotlib.pyplot as plt
from scripts.dec_tree import DecisionTree
from dec_bound import plot_decision_boundary


if __name__ == "__main__":
    
    data_path = "data/Dbig.txt"
    data = np.loadtxt(data_path) 
    samp_sizes = [32, 128, 512, 2048, 8192]
    perm = np.random.permutation(10000)
    train_data = data[perm[:8192],:]
    test_data = data[perm[8192:],:]
    
    err_percent = np.empty(len(samp_sizes))
    num_nodes = np.empty(len(samp_sizes))

    for (i,n) in enumerate(samp_sizes):

        #Train decision tree
        tree = DecisionTree(train_data[:n,])
        tree.train(train_data[:n,])

        #Count number of nodes in trained tree
        num_nodes[i] = tree.count_nodes()

        #Measure test set error
        preds = tree.predict(test_data[:,:2])
        errors = preds != test_data[:,2]
        err_percent[i] = np.mean(errors)

        #Plot decision boundary
        plot_decision_boundary(tree, test_data, f"Dbig{n}")

    print("Error percentages are", err_percent)
    print("Number of nodes are", num_nodes)

    plt.plot(samp_sizes, err_percent, marker="x")
    plt.xlabel("Training size")
    plt.ylabel("Error rate")
    plt.savefig("figs/learning_curve.png")
    plt.clf()
    
    plt.plot(samp_sizes, num_nodes, marker="x")
    plt.xlabel("Training size")
    plt.ylabel("Size of tree")
    plt.savefig("figs/complexity.png")











