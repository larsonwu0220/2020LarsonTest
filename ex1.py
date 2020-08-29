from sklearn.neural_network import MLPClassifier
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np

X, y = make_classification(n_samples=100,n_features=2, n_redundant=0, n_informative=2,n_clusters_per_class=1)
X_train, X_test, y_train, y_test = train_test_split(X, y, stratify=y,random_state=1)
print(X_train)
print(y_train)
plt.subplot(2,2,1)
ind0 = [index for index, value in enumerate(y_train) if value == 0]
ind1 = [index for index, value in enumerate(y_train) if value == 1]
plt.scatter(X_train[ind0,0],X_train[ind0,1],color='b', marker='o')
plt.scatter(X_train[ind1,0],X_train[ind1,1],color='r', marker='o')

plt.subplot(2,2,2)
plt.scatter(X_test[:,0],X_test[:,1],color = 'k', marker = '*')

### Training
clf = MLPClassifier(random_state=1, max_iter=300).fit(X_train, y_train)

### input testing data
y_test_pred=clf.predict_proba(X_test)
print(y_test_pred.shape)
print(y_test_pred)

ind0_pred = []
ind1_pred = []
for index, v in enumerate(y_test_pred):
    if v[0] > v[1]:
        ind0_pred.append(index)
    else:
        ind1_pred.append(index)

plt.subplot(2,2,3)
plt.scatter(X_test[ind0_pred,0],X_test[ind0_pred,1],color='b', marker='*')
plt.scatter(X_test[ind1_pred,0],X_test[ind1_pred,1],color='r', marker='*')
plt.show()

