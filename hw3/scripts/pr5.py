import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

y_test = np.array([1,1,0,1,1,0,1,1,0,0])
y_pred_prob = np.array([0.95, 0.85, 0.8, 0.7, 0.55, 0.45, 0.4, 0.3, 0.2, 0.1]) 
fpr, tpr, _ = metrics.roc_curve(y_test,  y_pred_prob)
print(tpr)
print(fpr)
print(_)

plt.plot(fpr,tpr)
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()
