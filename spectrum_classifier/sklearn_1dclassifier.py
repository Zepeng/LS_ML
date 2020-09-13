import warnings

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import numpy as np

import pickle
import uproot as up
from scipy.special import softmax
import random
import load_calib
import load_egamma

#Choose the training type
## FakeTest classify two type of fake data sin(x) and 2*x/pi
## CalibTrain train with calibration neutron and gamma
FakeTest = False
CalibTrain = False
EGammaTrain = True

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
    neutron_events = load_calib.LoadCalib('/junofs/users/lirh/DYB/run67527.root', 'FastNeutron')
    gamma_events = load_calib.LoadCalib('/junofs/users/lirh/DYB/run67522.root', 'CoTree')
    gamma_events = gamma_events[:2*len(neutron_events)]
    #Tag the events with 0 for neutron and 1 for gamma
    neutron_tags = np.zeros(len(neutron_events))
    gamma_tags = np.ones(len(gamma_events))
    #merge neutron and gamma datasets to make one single dataset.
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

if EGammaTrain == True:
    events = load_egamma.LoadEGamma('/hpcfs/juno/junogpu/luoxj/Data_PSD/elecsim_SumAllPmtWaves.root')
    print(events[0])
    indices = np.arange(len(events))
    random.shuffle(indices)
    #split the dataset into two parts for training and testing respectively.
    for index in indices:
        if len(input_train) < 0.8*len(events):
            input_train.append(events[index][0])
            target_train.append(events[index][1])
        else:
            input_test.append(events[index][0])
            target_test.append(events[index][1])

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
    predict_proba = classifier.predict_proba(input_test)
    print(predict_proba.shape)
    print(classifier.predict(input_test[:10]), target_test[:10])
    tag_0 = []
    tag_1 = []
    for i in range(len(predict_proba)):
        if target_test[i] == 0:
            tag_0.append(predict_proba[i][1])
        else:
            tag_1.append(predict_proba[i][1])
    fig1, ax1 = plt.subplots()
    n0, bins0, patches0 = ax1.hist(tag_0, bins=np.linspace(0, 1, 21), color='red', histtype='step',label='Neutron')
    n1, bins1, patches1 = ax1.hist(tag_1, bins=np.linspace(0, 1, 21), color='blue', histtype='step', label='Co')
    ax1.set_xlim(0,1)
    ax1.legend()
    ax1.set_xlabel('Prediction output')
    fig1.savefig('predicts.png')

    eff_neutron = []
    eff_gamma = []
    for i in range(len(n0)):
        eff_neutron.append(np.sum(n0[i:])*1.0/np.sum(n0))
        eff_gamma.append(np.sum(n1[i:])*1.0/np.sum(n1))
    fig2, ax2 = plt.subplots()
    ax2.plot(eff_neutron, eff_gamma)
    ax2.set_xlabel('Neutron efficiency')
    ax2.set_ylabel('Gamma efficiency')
    ax2.set_xlim(0, 0.3)
    ax2.set_ylim(0, 1)
    fig2.savefig('roc.png')

print("Training set score: %f" % classifier.score(input_train, target_train))
print("Test set score: %f" % classifier.score(input_test, target_test))
