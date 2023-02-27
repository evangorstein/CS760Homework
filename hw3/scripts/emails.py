import numpy as np
import pandas as pd
from scipy.special import expit
from sklearn.neighbors import KNeighborsClassifier
from sklearn import metrics
import matplotlib.pyplot as plt


def custom_cv_folds(X, k):
    """
    Produces list of length k arrays
    Each entry in list has the indices of the test set for 
    one of the k cross validation splits
    """
    n = X.shape[0]
    cv_ids = []
    for i in range(k):
        id = np.arange(n * i / k, n * (i+1) / k, dtype=int)
        cv_ids.append(id)
    
    return cv_ids


class LogiClassifier():

    def __init__(self) -> None:
        pass

    def fit(self, X, y, init_beta = None, lr = .1, max_iter = 1000, thres = 1e-6):

        n, p = X.shape
        X = np.hstack([np.ones((n,1)), X])
        
        if init_beta is None:
            init_beta = np.zeros(p+1)
        
        beta = np.copy(init_beta)
        for i in range(int(max_iter)):
            if i % 100 == 0:
                print(f"At iteration {i}")

            lin_pred = X @ beta   
            resid = y - expit(lin_pred)
            neg_grad = X.T @ resid/n

            #If gradient is small, exit
            scaled_norm = np.linalg.norm(neg_grad)/n
            if (scaled_norm < thres):
                break

            beta = beta + lr*neg_grad
            
        self.pars = beta

        return None 

    def predict(self, X):

        n = X.shape[0]
        return expit(np.hstack([np.ones((n,1)), X]) @ self.pars)



if __name__ == '__main__':

    #Read data
    data = pd.read_csv("../data/emails.csv")
    X = data.drop(["Email No.","Prediction"], axis=1)
    y = data["Prediction"]

    n_folds = 5
    #Get fold indices
    ids = custom_cv_folds(X, n_folds)

    #------Get cross validation predictions for logistic regression----------------#
    log_cv_preds = []
    for id in ids:
        logi = LogiClassifier()
        X_train = X.drop(id)
        y_train = y.drop(id)
        logi.fit(X_train, y_train)

        X_test = X.iloc[id]
        preds = logi.predict(X_test)
        log_cv_preds.append(preds)


    #------Get cross validation predictions for k nearest neighbors----------------#
    num_neighs = [1,3,5,7,10] #k's
    #Storage of predictions of each model in each fold
    knn_all_preds = []
    for k in num_neighs:

        #Storage of predictions of a single model in each fold
        knn_cv_preds = []
        for id in ids:
            
            neigh = KNeighborsClassifier(n_neighbors = k)

            X_train = X.drop(id)
            y_train = y.drop(id)
            neigh.fit(X_train, y_train)

            X_test = X.iloc[id]
            preds = neigh.predict(X_test)

            knn_cv_preds.append(preds) 
        
        knn_all_preds.append(knn_cv_preds)

    #---------Report results for each fold for 1NN and logistic regression------------#
    cv_knn_acc, cv_knn_prec, cv_knn_recall = [], [], []
    cv_log_acc, cv_log_prec, cv_log_recall = [], [], []

    for (id, knn_preds, log_preds) in zip(ids, knn_all_preds[0], log_cv_preds):
        
        #Get test set true labels
        true_vals = y.iloc[id]

        #Get predicted values from logistic model by rounding
        log_preds = np.round(log_preds)
        #Metrics for logistic model
        log_acc = np.mean(true_vals == log_preds)
        cv_log_acc.append(log_acc)
        log_prec = np.sum(true_vals*log_preds)/np.sum(log_preds)
        cv_log_prec.append(log_prec)
        log_recall =  np.sum(true_vals*log_preds)/np.sum(true_vals)
        cv_log_recall.append(log_recall)

        #Metrics for knn (with k=1)
        knn_acc = np.mean(true_vals == knn_preds)
        cv_knn_acc.append(knn_acc)
        knn_prec = np.sum(true_vals*knn_preds)/np.sum(knn_preds)
        cv_knn_prec.append(knn_prec)
        knn_recall =  np.sum(true_vals*knn_preds)/np.sum(true_vals)
        cv_knn_recall.append(knn_recall)

    print(f"Accuracy from 1NN in folds was {cv_knn_acc}\n")
    print(f"Precision from 1NN in folds was {cv_knn_prec}\n")
    print(f"Recall from 1NN in folds was {cv_knn_recall}\n")

    print(f"Accuracy from logistic reg in folds was {cv_log_acc}\n")
    print(f"Precision from logistic reg in folds was {cv_log_prec}\n")
    print(f"Recall from logistic reg in folds was {cv_log_recall}\n")
    
    #------Report average (across folds) accuracy of k-nearest neighbors for all k's-----------#
    all_acc = []
    for cv_preds in knn_all_preds:
        
        #Concatenate predictions from all 5 folds into single vector
        single_list_preds = np.concatenate(cv_preds)
        acc = np.mean(y == single_list_preds)
        all_acc.append(acc)
    
    print(f"Accuracy of knn for various k's was {all_acc}\n")

    plt.plot(num_neighs, all_acc, marker='o', color = "red")
    plt.ylabel("Average accuracy")
    plt.yticks(np.linspace(0.835, 0.855, 5))
    plt.xlabel("k")
    plt.xticks(np.linspace(2,10,5))
    plt.grid(True)
    plt.title("kNN 5 fold cross validation")
    plt.show()


    #-----------------Single train-test split analysis-------------------------------------------#
    np.random.seed(420770)
    indices = np.random.permutation(X.shape[0])
    training_idx, test_idx = indices[:4000], indices[4000:]
    X_train, X_test = X.iloc[training_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[training_idx], y.iloc[test_idx]

    # Logistic regression
    logi = LogiClassifier()
    logi.fit(X_train, y_train, max_iter = 3000)
    logi_probs = logi.predict(X_test)

    # K Nearest Neighbors
    neigh = KNeighborsClassifier(n_neighbors = 5)
    neigh.fit(X_train, y_train)
    neigh_probs = neigh.predict_proba(X_test)[:,1]

    #Get metrics 
    fpr_log, tpr_log, _ = metrics.roc_curve(y_test,  logi_probs)
    auc_log = metrics.roc_auc_score(y_test, logi_probs)
    fpr_neigh, tpr_neigh, _ = metrics.roc_curve(y_test,  neigh_probs)
    neigh_log = metrics.roc_auc_score(y_test, neigh_probs)

    #Plot
    plt.plot(fpr_log,tpr_log, color = "orange", label = f"LogisticRegression (AUC = {auc_log:.2f})")
    plt.plot(fpr_neigh, tpr_neigh, color = "blue", label = f"KNearestNeighbors (AUC = {neigh_log:.2f})") 
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.legend(loc="lower right")
    plt.show()





