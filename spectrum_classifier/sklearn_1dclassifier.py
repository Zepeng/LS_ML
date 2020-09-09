import warnings

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import numpy as np

import pickle
import uproot as up
from scipy.special import softmax
import random

#load the data from root file.
def loadroot(rootfile):
    tfile = uproot.open(rootfile)

def loadfakedata(datafile):
    dataset = np.load(datafile, allow_pickle=True)
    #a debug print to screen to check the data loading is correct
    if False:
        print(dataset[0][0], dataset[0][1])
    return dataset

dataset = loadfakedata('fake_data.npy')
random.shuffle(dataset)
input_train = []
target_train = []
input_test = []
target_test = []
#split the dataset into train and test set.
for i in range(len(dataset)):
    if i < 0.8*len(dataset):
        input_train.append(dataset[i][0])
        target_train.append(dataset[i][1])
    else:
        input_test.append(dataset[i][0])
        target_test.append(dataset[i][1])

#use a ML model from scikit
from sklearn import naive_bayes
classifier = naive_bayes.GaussianNB()
#train the network
# warning and are ignore it here
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category=ConvergenceWarning,
                            module="sklearn")
    classifier.fit(input_train, target_train)
    pkl_filename = 'pickle_model.pkl'
    with open(pkl_filename, 'wb') as pfile:
        pickle.dump(classifier, pfile)

if True:
    predict_proba = classifier.predict_proba(input_test[:10])
    targets = target_test[:10]
    print(classifier.predict(input_test[:10]), targets)
    for item, target in zip(predict_proba, targets):
        print(item, softmax(item), target)
print("Training set score: %f" % classifier.score(input_train, target_train))
print("Test set score: %f" % classifier.score(input_test, target_test))
