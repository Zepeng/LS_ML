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
        #This module is designed to do only three types to task.
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

    def loadDSNB(self, datafile:str):
        dataset = np.load(datafile, allow_pickle=True)
        data_sig_NoweightE = dataset["sig"][:, 0]
        data_bkg_NoweightE = dataset["bkg"][:, 0]
        data_sig_weightE = dataset["sig"][:, 1]
        data_bkg_weightE = dataset["bkg"][:, 1]

        data_sig = np.concatenate((data_sig_NoweightE, data_sig_weightE),axis=1)
        data_bkg = np.concatenate((data_bkg_NoweightE, data_bkg_weightE),axis=1)

        print(f"len(data_sig):{len(data_sig)}, len(data_bkg):{len(data_bkg)}")
        label_sig = np.ones(len(data_sig), dtype=np.int)
        label_bkg = np.zeros(len(data_bkg), dtype=np.int)
        data_return = np.vstack((data_sig, data_bkg))
        label_return = np.concatenate((label_sig, label_bkg))
        print(f"shape of data :{data_return.shape}")
        print(f"labels : {label_return}")
        if len(data_return)!=len(label_return):
            print("There existing a Error cause length of labels and data don't match !!!!!")
            exit(1)

        #shuffle the signal and background events by indices
        input_train = []
        target_train = []
        input_test = []
        target_test = []
        indices = np.arange(len(data_return))
        random.shuffle(indices)
        #split the dataset into two parts for training and testing respectively.
        for index in indices:
            if len(self.input_train) < 0.8*len(data_return):
                self.input_train.append(data_return[index])
                self.target_train.append(label_return[index])
            else:
                self.input_test.append(data_return[index])
                self.target_test.append(label_return[index])
        # return (data_return, label_return)

    #load the dataset for gamma/neutron separation from calibration
    def loadcalib(self, neutronfile, gammafile):
        #load the data from root file.
        neutron_events = load_calib.LoadCalib(neutronfile, 'FastNeutron')
        gamma_events = load_calib.LoadCalib(gammafile, 'CoTree')
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
        input_train = []
        for index in indices:
            if len(input_train) < 0.8*len(events):
                self.input_train.append(events[index])
                self.target_train.append(tags[index])
            else:
                self.input_test.append(events[index])
                self.target_test.append(tags[index])

    def loadegamma(self, rootfile):
        #Load the e/gamma spectrum from rootfile
        events = load_egamma.LoadEGamma(rootfile)
        print(events[0])
        indices = np.arange(len(events))
        random.shuffle(indices)
        #split the dataset into two parts for training and testing respectively.
        input_train = []
        for index in indices:
            if len(input_train) < 0.8*len(events):
                self.input_train.append(events[index][0])
                self.target_train.append(events[index][1])
            else:
                self.input_test.append(events[index][0])
                self.target_test.append(events[index][1])

    def AddModels(self):
        #Add multiple models for comparison, look for the definition\
        #of each model on sklearn documentation
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC, LinearSVC
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.tree import DecisionTreeClassifier
        from sklearn.ensemble import RandomForestClassifier
        from sklearn.naive_bayes import GaussianNB
        from sklearn.neural_network import MLPClassifier
        from sklearn.naive_bayes import GaussianNB

        models = {}

        # models["LogisticRegression"]  = LogisticRegression()
        # models["SVC"] = SVC()
        # models["LinearSVC"] = LinearSVC()
        # models["KNeighbors"] = KNeighborsClassifier()
        # models["DecisionTree"] = DecisionTreeClassifier()
        # models["RandomForest"] = RandomForestClassifier()
        # rf2 = RandomForestClassifier(n_estimators=100, criterion='gini',
        #                                         max_depth=10, random_state=0, max_features=None)
        # models["RandomForest2"] = rf2
        models["MLPClassifier"] = MLPClassifier(solver='lbfgs', random_state=0)
        # models["GaussianNB"] =  GaussianNB()
        self.models = models

    def CompareModels(self):
        #compare the efficiency of different models.
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
        #Train and save a specific model.
        if modelname not in self.models.keys():
            print('Model not supported yet!')
            sys.exit()
        else:
            print('Train model %s' % modelname)
            classifier = self.models[modelname]
            classifier.fit(self.input_train, self.target_train)
            pkl_filename = 'pickle_model.pkl'
            with open(pkl_filename, 'wb') as pfile:
                pickle.dump(classifier, pfile)
            input_test = self.input_test
            target_test = self.target_test
            if True:
                predict_proba = classifier.predict_proba(input_test)
                print(predict_proba.shape)
                print(classifier.predict(input_test[:100]), target_test[:100])
    def LoadValidateData(self, name_file):
        dataset = np.load(name_file, allow_pickle=True)
        # data_sig = dataset["sig"][:, 1]
        # data_bkg = dataset["bkg"][:, 1]
        data_sig_NoweightE = dataset["sig"][:, 0]
        data_bkg_NoweightE = dataset["bkg"][:, 0]
        data_sig_weightE = dataset["sig"][:, 1]
        data_bkg_weightE = dataset["bkg"][:, 1]

        data_sig = np.concatenate((data_sig_NoweightE, data_sig_weightE),axis=1)
        data_bkg = np.concatenate((data_bkg_NoweightE, data_bkg_weightE),axis=1)

        print(f"len(data_sig):{len(data_sig)}, len(data_bkg):{len(data_bkg)}")

        n_validate = len(data_bkg)
        label_sig = np.ones(len(data_sig), dtype=np.int)
        label_bkg = np.zeros(len(data_bkg), dtype=np.int)
        self.input_validate = np.vstack((data_sig[:n_validate], data_bkg[:n_validate]))
        self.label_validate = np.concatenate((label_sig[:n_validate], label_bkg[:n_validate]))
        print(f"shape of data :{self.input_validate.shape}")
        print(f"labels : {self.label_validate}")
        if len(self.input_validate)!=len(self.label_validate):
            print("There existing a Error cause length of labels and data don't match !!!!!")
            exit(1)

    def LoadModelPredict(self, name_file_model):
        n_correct = 0
        n_correct_sig = 0
        n_correct_bkg = 0
        n_wrong_into_sig = 0
        n_wrong_into_bkg = 0

        #Load Model and do the predict
        with open( name_file_model, 'rb') as fr:
            new_svm = pickle.load(fr)

            for i in range(len(self.input_validate)):
                prediction = new_svm.predict(self.input_validate[i].reshape((1, len(self.input_validate[i]))))
                if prediction == self.label_validate[i]:
                    n_correct +=1
                    if self.label_validate[i]==1:
                        n_correct_sig += 1
                    else:
                        n_correct_bkg += 1
                else:
                    if self.label_validate[i]==1:
                        n_wrong_into_bkg += 1
                    else:
                        n_wrong_into_sig += 1
                print(f"Prediction : {prediction}, Anwser : {self.label_validate[i]}")
        print(f"Score : {n_correct/len(self.input_validate)} [{n_correct}/{len(self.input_validate)}]")
        print(f"correct_sig : {n_correct_sig/len(self.label_validate)*2}   [{n_correct_sig}/{len(self.label_validate)/2}]")
        print(f"correct_bkg : {n_correct_bkg/len(self.label_validate)*2}   [{n_correct_bkg}/{len(self.label_validate)/2}]")
        print(f"wrong into signal : {n_wrong_into_sig/len(self.label_validate)*2} [{n_wrong_into_sig}/{len(self.label_validate)/2}]")
        print(f"wrong into bkg    : {n_wrong_into_bkg/len(self.label_validate)*2} [{n_wrong_into_bkg}/{len(self.label_validate)/2}]")
    def LoadModelGetEfficiency(self, name_file_model):
        with open( name_file_model, 'rb') as fr:
            classifier = pickle.load(fr)
        predict_proba = classifier.predict_proba(self.input_validate)
        tag_0 = []
        tag_1 = []
        for i in range(len(predict_proba)):
            if self.label_validate[i] == 0:
                tag_0.append(predict_proba[i][1])
            else:
                tag_1.append(predict_proba[i][1])
        fig1, ax1 = plt.subplots()
        n0, bins0, patches0 = ax1.hist(tag_0, bins=np.linspace(0, 1, 100), color='red', histtype='step',label='Background')
        n1, bins1, patches1 = ax1.hist(tag_1, bins=np.linspace(0, 1, 100), color='blue', histtype='step', label='Signal')
        ax1.set_xlim(0,1)
        plt.semilogy()
        ax1.legend()
        ax1.set_xlabel('Prediction output')
        fig1.savefig('predicts.png')

        eff_bkg = []
        eff_sig = []
        print(f"n0: {n0}, \n n1: {n1}")
        for i in range(len(n0)):
            eff_bkg.append(np.sum(n0[i:])*1.0/np.sum(n0))
            eff_sig.append(np.sum(n1[i:])*1.0/np.sum(n1))
        fig2, ax2 = plt.subplots()
        ax2.plot(eff_bkg, eff_sig)
        ax2.set_xlabel('Background efficiency')
        ax2.set_ylabel('Signal efficiency')
        ax2.set_xlim(0, 0.1)
        ax2.set_ylim(0, 1)
        fig2.savefig('roc.png')

if __name__ == '__main__':
    ana = SpectrumAna('FakeTest')

    # ana.loadDSNB("./try.npz")
    # ana.loadDSNB("./traindata_extendedtime.npz")
    # ana.AddModels()
    # ana.CompareModels()
    # ana.TrainModel('MLPClassifier')

    ana.LoadValidateData("test.npz")
    # ana.LoadModelPredict("pickle_model.pkl")
    ana.LoadModelGetEfficiency("model_combie.pkl")
    plt.show()

