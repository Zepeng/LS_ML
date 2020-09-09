import warnings

import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.exceptions import ConvergenceWarning
import numpy as np

import pickle

dataset = np.load('fake_data.npy')
X_train = []
X_test = []
y_train = []
y_test = []
#a debug print to screen to check the data loading is correct
if False:
    print(dataset[0][0], dataset[0][1])
#split the dataset into train and test set.
for i in range(len(dataset)):
    if i < 0.8*len(dataset):
        X_train.append(dataset[i][0])
        y_train.append(dataset[i][1])
    else:
        X_test.append(dataset[i][0])
        y_test.append(dataset[i][1])

#use a ML model from scikit
from sklearn import naive_bayes
classifier = naive_bayes.GaussianNB()
#train the network
# warning and are ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    classifier.fit(X_train, y_train)
    pkl_filename = 'pickle_model.pkl'
    with open(pkl_filename, 'wb') as pfile:
        pickle.dump(classifier, pfile)

print(classifier.predict_proba(X_test[:10]), y_test[:10])
print("Training set score: %f" % classifier.score(X_train, y_train))
print("Test set score: %f" % classifier.score(X_test, y_test))
