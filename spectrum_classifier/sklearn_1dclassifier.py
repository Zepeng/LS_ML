import warnings

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
import numpy as np

dataset = np.load('fake_data.npy')
X_train = []
X_test = []
y_train = []
y_test = []
print(dataset[0][0], dataset[0][1])
for i in range(len(dataset)):
    if i < 0.8*len(dataset):
        X_train.append(dataset[i][0])
        y_train.append(dataset[i][1])
    else:
        X_test.append(dataset[i][0])
        y_test.append(dataset[i][1])

from sklearn import naive_bayes
classifier = naive_bayes.GaussianNB()
# this example won't converge because of CI's time constraints, so we catch the
# warning and are ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    classifier.fit(X_train, y_train)

print(classifier.predict(X_test[:10]), y_test[:10])
print("Training set score: %f" % classifier.score(X_train, y_train))
print("Test set score: %f" % classifier.score(X_test, y_test))
