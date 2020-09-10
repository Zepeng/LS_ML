import warnings

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import numpy as np

import pickle
import uproot as up
from scipy.special import softmax
import random
import load_calib

#Choose the training type
## FakeTest classify two type of fake data sin(x) and 2*x/pi
## CalibTrain train with calibration neutron and gamma
FakeTest = False
CalibTrain = True

#load the fake data for test
def loadfakedata(datafile):
    dataset = np.load(datafile, allow_pickle=True)
    #a debug print to screen to check the data loading is correct
    if False:
        print(dataset[0][0], dataset[0][1])
    return dataset

#build the dataset for train/test
input_train = []
target_train = []
input_test = []
target_test = []

#load fake data for test.
if FakeTest == True:
    dataset = loadfakedata('fake_data.npy')
    random.shuffle(dataset)
    #split the dataset into train and test set.
    for i in range(len(dataset)):
        if i < 0.8*len(dataset):
            input_train.append(dataset[i][0])
            target_train.append(dataset[i][1])
        else:
            input_test.append(dataset[i][0])
            target_test.append(dataset[i][1])

#load calibration data
if CalibTrain ==True:
    #load the data from root file.
    neutron_events = load_calib.LoadCalib('/junofs/users/lirh/DYB/run67527.root', 'FastNeutron', 18937)
    neutron_tags = np.zeros(len(neutron_events))
    gamma_events = load_calib.LoadCalib('/junofs/users/lirh/DYB/run67522.root', 'CoTree', 18937)
    gamma_tags = np.ones(len(gamma_events))
    events = np.concatenate((neutron_events, gamma_events) , axis=0)
    tags = np.concatenate((neutron_tags, gamma_tags))
    #shuffle the signal and background events by indices
    indices = np.arange(len(events))
    random.shuffle(indices)
    #split the dataset into two parts for training and testing respectively.
    for index in indices:
        if len(input_train) < 0.8*len(events):
            input_train.append(events[index])
            target_train.append(tags[index])
        else:
            input_test.append(events[index])
            target_test.append(tags[index])

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
