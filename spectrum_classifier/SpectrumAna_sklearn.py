import warnings

import matplotlib.pyplot as plt
from sklearn.exceptions import ConvergenceWarning
import numpy as np

import pickle, sys
import uproot as up
from scipy.special import softmax
import random
import load_calib
import load_egamma

class SpectrumAna():
    TaskType = '' #choose a task type from faketest, calibtrain, egammatrain
    dataset = []
    TrainResult = []
    TestRestult = []
    input_train = []
    target_train = []
    input_test = []
    target_test = []
    models = []
    def __init__(self, task):
        if task not in ['FakeTest', 'CalibTrain', 'EGammaTrain']:
            print('The task is not supported by this module!')
            sys.exit()
        self.TaskType = task


    #load the fake data for test
    def loadfakedata(self, datafile):
        dataset = np.load(datafile, allow_pickle=True)
        #a debug print to screen to check the data loading is correct
        if False:
            print(dataset[0][0], dataset[0][1])
        random.shuffle(dataset)
        #split the dataset into train and test set.
        for i in range(len(dataset)):
            if i < 0.8*len(dataset):
                self.input_train.append(dataset[i][0])
                self.target_train.append(dataset[i][1])
            else:
                self.input_test.append(dataset[i][0])
                self.target_test.append(dataset[i][1])


    #load the dataset for gamma/neutron separation from calibration
    def loadcalib(self, neutronfile, gammafile):
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
                self.input_train.append(events[index])
                sefl.target_train.append(tags[index])
            else:
                self.input_test.append(events[index])
                self.target_test.append(tags[index])

    def loadegamma(self, rootfile):
        events = load_egamma.LoadEGamma('/hpcfs/juno/junogpu/luoxj/Data_PSD/elecsim_SumAllPmtWaves.root')
        print(events[0])
        indices = np.arange(len(events))
        random.shuffle(indices)
        #split the dataset into two parts for training and testing respectively.
        for index in indices:
            if len(input_train) < 0.8*len(events):
                self.input_train.append(events[index][0])
                self.target_train.append(events[index][1])
            else:
                self.input_test.append(events[index][0])
                self.target_test.append(events[index][1])

    def AddModels(self):
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        from sklearn.naive_bayes import GaussianNB

        models = {}

        models["LogisticRegression"]  = LogisticRegression()
        models["SVC"] = SVC()
        models["LinearSVC"] = LinearSVC()
        models["KNeighbors"] = KNeighborsClassifier()
        models["DecisionTree"] = DecisionTreeClassifier()
        models["RandomForest"] = RandomForestClassifier()
        rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
                                                max_depth=10, random_state=0, max_features=None)
        models["RandomForest2"] = rf2
        models["MLPClassifier"] = MLPClassifier(solver='lbfgs', random_state=0)
        models["GaussianNB"] =  GaussianNB()
        self.models = models

    def CompareModels(self):
        results = []
        names = []
        models = self.models
        from sklearn.model_selection import cross_val_score
        for name in models.keys():
            model = models[name]
            result = cross_val_score(model, self.input_train, self.target_train)
            names.append(name)
            results.append(result)
        for i in range(len(names)):
            print(names[i],results[i].mean())

    def TrainModel(self, modelname):
        if modelname not in self.models.keys():
            print('Model not supported yet!')
            sys.exit()
        else:
            print('Train model %s' % modelname)
            classifier = self.models[modelname]
            classifier.fit(input_train, target_train)
            pkl_filename = 'pickle_model.pkl'
            with open(pkl_filename, 'wb') as pfile:
                pickle.dump(classifier, pfile)

ana = SpectrumAna('FakeTest')
ana.loadfakedata('./fake_data.npy')
ana.AddModels()
ana.CompareModels()
ana.TrainModel('MLPClassifier')
"""

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
"""
