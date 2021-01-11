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
import pandas as pd


class SpectrumAna():
    TaskType = ''  # choose a task type from faketest, calibtrain, egammatrain
    dataset = []
    TrainResult = []
    TestRestult = []
    input_train = []
    target_train = []
    input_test = []
    target_test = []
    models = []

    def __init__(self, task):
        # This module is designed to do only three types to task.
        if task not in ['FakeTest', 'CalibTrain', 'EGammaTrain']:
            print('The task is not supported by this module!')
            sys.exit()
        self.TaskType = task

    # load the fake data for test
    def loadfakedata(self, datafile):
        dataset = np.load(datafile, allow_pickle=True)
        # a debug print to screen to check the data loading is correct
        if False:
            print(dataset[0][0], dataset[0][1])
        random.shuffle(dataset)
        # split the dataset into train and test set.
        for i in range(len(dataset)):
            if i < 0.8 * len(dataset):
                self.input_train.append(dataset[i][0])
                self.target_train.append(dataset[i][1])
            else:
                self.input_test.append(dataset[i][0])
                self.target_test.append(dataset[i][1])

    def LoadFileListIntoDataset(self, filelist:list):
        if len(filelist) == 0:
            print("ERROR!!! Input filelist length is zero !!! in function 'LoadFileListIntoDataset()'")
        dataset = np.load(filelist[0], allow_pickle=True)
        dataset_return = {}
        print(f"key : {dataset.files}")
        for key in dataset.files:
            dataset_return[key] = dataset[key]
        for file in filelist[1:]:
            dataset_tmp = np.load(file, allow_pickle=True)
            def inner(dataset_tmp):
                for key in dataset.files:
                    if len(dataset_tmp[key]) == 0:
                        return
                for key in dataset.files:
                    dataset_return[key] = np.concatenate((dataset_return[key], dataset_tmp[key]))
            inner(dataset_tmp)
        print(f"Check shapes of dataset : ")
        for key in dataset.files:
            print(f"{key} -> {dataset_return[key].shape}")
        return dataset_return
    def loadDSNB(self, datafileslist, i_scheme=0):
        # dataset = np.load(datafile, allow_pickle=True)
        dataset = self.LoadFileListIntoDataset(datafileslist)
        data_sig_NoweightE = dataset["sig"][:, 0]
        data_bkg_NoweightE = dataset["bkg"][:, 0]
        data_sig_weightE = dataset["sig"][:, 1]
        data_bkg_weightE = dataset["bkg"][:, 1]
        # dataset_sig = dataset["sig"].item()
        # dataset_bkg = dataset["bkg"].item()
        # data_sig_NoweightE = dataset_sig["NoWeightE"]
        # data_bkg_NoweightE = dataset_bkg["NoWeightE"]
        # data_sig_weightE = dataset_sig["WeightE"]
        # data_bkg_weightE = dataset_bkg["WeightE"]
        data_sig_vertex = dataset["sig_vertex"]
        data_bkg_vertex = dataset["bkg_vertex"]
        data_sig_equen = dataset["sig_equen"]
        data_bkg_equen = dataset["bkg_equen"]
        data_bkg_pdg = dataset["bkg_pdg"]
        data_bkg_px = dataset["bkg_px"]
        data_bkg_py = dataset["bkg_py"]
        data_bkg_pz = dataset["bkg_pz"]

        (data_sig_weightE, data_sig_NoweightE, data_bkg_weightE, data_bkg_NoweightE, data_sig_equen,
         data_bkg_equen, data_sig_vertex, data_bkg_vertex, data_bkg_pdg,
                data_bkg_px, data_bkg_py, data_bkg_pz ) = \
            self.VertexCut(data_sig_vertex, data_bkg_vertex, data_sig_weightE, data_bkg_weightE, data_sig_NoweightE,
                           data_bkg_NoweightE, data_sig_equen, data_bkg_equen, data_bkg_pdg, data_bkg_px,
                           data_bkg_py, data_bkg_pz, i_scheme)

        # data_sig = np.concatenate((data_sig_NoweightE, data_sig_weightE, data_sig_vertex/10000, data_sig_equen.reshape(len(data_sig_equen), 1)),axis=1)
        # data_bkg = np.concatenate((data_bkg_NoweightE, data_bkg_weightE, data_bkg_vertex/10000, data_bkg_equen.reshape(len(data_bkg_equen), 1)),axis=1)

        ###################combine two scheme######################################
        data_sig = np.concatenate((data_sig_NoweightE, data_sig_weightE), axis=1)
        data_bkg = np.concatenate((data_bkg_NoweightE, data_bkg_weightE), axis=1)
        ###########################################################################

        # data_sig = data_sig_NoweightE
        # data_bkg = data_bkg_NoweightE

        print(f"len(data_sig) Before align length:{len(data_sig)}, len(data_bkg):{len(data_bkg)}")
        len_input = np.min([len(data_sig), len(data_bkg)])
        data_sig, data_bkg = data_sig[:len_input], data_bkg[:len_input]
        print(f"len(data_sig) After align length:{len(data_sig)}, len(data_bkg):{len(data_bkg)}")
        label_sig = np.ones(len(data_sig), dtype=np.int)
        label_bkg = np.zeros(len(data_bkg), dtype=np.int)
        data_return = np.vstack((data_sig, data_bkg))
        label_return = np.concatenate((label_sig, label_bkg))
        print(f"shape of data :{data_return.shape}")
        print(f"labels : {label_return}")
        if len(data_return) != len(label_return):
            print("There existing a Error cause length of labels and data don't match !!!!!")
            exit(1)

        # shuffle the signal and background events by indices
        input_train = []
        target_train = []
        input_test = []
        target_test = []
        indices = np.arange(len(data_return))
        random.shuffle(indices)
        # split the dataset into two parts for training and testing respectively.
        for index in indices:
            if len(self.input_train) < 0.8 * len(data_return):
                self.input_train.append(data_return[index])
                self.target_train.append(label_return[index])
            else:
                self.input_test.append(data_return[index])
                self.target_test.append(label_return[index])
        # return (data_return, label_return)

    # load the dataset for gamma/neutron separation from calibration
    def loadcalib(self, neutronfile, gammafile):
        # load the data from root file.
        neutron_events = load_calib.LoadCalib(neutronfile, 'FastNeutron')
        gamma_events = load_calib.LoadCalib(gammafile, 'CoTree')
        gamma_events = gamma_events[:2 * len(neutron_events)]
        # Tag the events with 0 for neutron and 1 for gamma
        neutron_tags = np.zeros(len(neutron_events))
        gamma_tags = np.ones(len(gamma_events))
        # merge neutron and gamma datasets to make one single dataset.
        events = np.concatenate((neutron_events, gamma_events), axis=0)
        tags = np.concatenate((neutron_tags, gamma_tags))
        # shuffle the signal and background events by indices
        indices = np.arange(len(events))
        random.shuffle(indices)
        # split the dataset into two parts for training and testing respectively.
        input_train = []
        for index in indices:
            if len(input_train) < 0.8 * len(events):
                self.input_train.append(events[index])
                self.target_train.append(tags[index])
            else:
                self.input_test.append(events[index])
                self.target_test.append(tags[index])

    def loadegamma(self, rootfile):
        # Load the e/gamma spectrum from rootfile
        events = load_egamma.LoadEGamma(rootfile)
        print(events[0])
        indices = np.arange(len(events))
        random.shuffle(indices)
        # split the dataset into two parts for training and testing respectively.
        input_train = []
        for index in indices:
            if len(input_train) < 0.8 * len(events):
                self.input_train.append(events[index][0])
                self.target_train.append(events[index][1])
            else:
                self.input_test.append(events[index][0])
                self.target_test.append(events[index][1])

    def AddModels(self):
        # Add multiple models for comparison, look for the definition\
        # of each model on sklearn documentation
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
        # compare the efficiency of different models.
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
            print(names[i], results[i].mean())

    def TrainModel(self, modelname, pkl_filename='pickle_model.pkl' ):
        # Train and save a specific model.
        if modelname not in self.models.keys():
            print('Model not supported yet!')
            sys.exit()
        else:
            print('Train model %s' % modelname)
            classifier = self.models[modelname]
            classifier.fit(self.input_train, self.target_train)
            with open(pkl_filename, 'wb') as pfile:
                pickle.dump(classifier, pfile)
            input_test = self.input_test
            target_test = self.target_test
            if True:
                predict_proba = classifier.predict_proba(input_test)
                print(predict_proba.shape)
                print(classifier.predict(input_test[:100]), target_test[:100])

    def VertexCut(self, data_sig_vertex, data_bkg_vertex, data_sig_weightE, data_bkg_weightE, data_sig_NoweightE,
                  data_bkg_NoweightE, data_sig_Equen, data_bkg_Equen, data_bkg_pdg, data_bkg_px,
                  data_bkg_py, data_bkg_pz, i_scheme=0):
        ####make a vertex cut for dataset###############
        data_sig_R = np.sqrt(np.sum(data_sig_vertex ** 2, axis=1)) / 1000
        data_bkg_R = np.sqrt(np.sum(data_bkg_vertex ** 2, axis=1)) / 1000
        # print(f"shape of data_sig_R : {data_sig_R.shape}, shape of data_sig_vertex: {data_sig_vertex.shape}")

        if (i_scheme == 0):
            cut_indices_sig = (data_sig_R ** 3 < 4096)
            cut_indices_bkg = (data_bkg_R ** 3 < 4096)
        elif (i_scheme == 1):
            cut_indices_sig = (data_sig_R ** 3 < 1000)
            cut_indices_bkg = (data_bkg_R ** 3 < 1000)
        elif (i_scheme == 2):
            cut_indices_sig = ((data_sig_R ** 3 >= 1000) & (data_sig_R ** 3 < 2000))
            cut_indices_bkg = ((data_bkg_R ** 3 >= 1000) & (data_bkg_R ** 3 < 2000))
        elif (i_scheme == 3):
            cut_indices_sig = (data_sig_R ** 3 >= 2000) & (data_sig_R ** 3 < 3000)
            cut_indices_bkg = (data_bkg_R ** 3 >= 2000) & (data_bkg_R ** 3 < 3000)
        elif (i_scheme == 4):
            cut_indices_sig = (data_sig_R ** 3 >= 3000) & (data_sig_R ** 3 < 4096)
            cut_indices_bkg = (data_bkg_R ** 3 >= 3000) & (data_bkg_R ** 3 < 4096)
        else:
            print("Wrong i_scheme input!!!!")
            exit(1)

        if Equen_cut == True:
            Ecut_indices_sig = ((data_sig_Equen<31) & (data_sig_Equen>10))
            Ecut_indices_bkg = ((data_bkg_Equen<31) & (data_bkg_Equen>10))
            cut_indices_sig = (cut_indices_sig & Ecut_indices_sig)
            cut_indices_bkg = (cut_indices_bkg & Ecut_indices_bkg)

        data_sig_weightE, data_sig_NoweightE = data_sig_weightE[cut_indices_sig], data_sig_NoweightE[cut_indices_sig]
        data_bkg_weightE, data_bkg_NoweightE = data_bkg_weightE[cut_indices_bkg], data_bkg_NoweightE[cut_indices_bkg]
        data_sig_Equen, data_bkg_Equen = data_sig_Equen[cut_indices_sig], data_bkg_Equen[cut_indices_bkg]
        data_sig_vertex, data_bkg_vertex = data_sig_vertex[cut_indices_sig], data_bkg_vertex[cut_indices_bkg]

        if study_pdg:
            data_bkg_pdg = data_bkg_pdg[cut_indices_bkg]
            data_bkg_px, data_bkg_py, data_bkg_pz = data_bkg_px[cut_indices_bkg], data_bkg_py[cut_indices_bkg], data_bkg_pz[cut_indices_bkg]
        else:
            data_bkg_R = []
            data_bkg_px, data_bkg_py, data_bkg_pz = [], [], []
        print(f"check shape input : {data_bkg_weightE.shape}")
        #################################################
        return (data_sig_weightE, data_sig_NoweightE, data_bkg_weightE, data_bkg_NoweightE,
                data_sig_Equen, data_bkg_Equen, data_sig_vertex, data_bkg_vertex, data_bkg_pdg,
                data_bkg_px, data_bkg_py, data_bkg_pz)

    def LoadValidateData(self, datafileslist, i_scheme=0):
        # dataset = np.load(name_file, allow_pickle=True)
        dataset = self.LoadFileListIntoDataset(datafileslist)
        # data_sig = dataset["sig"][:, 1]
        # data_bkg = dataset["bkg"][:, 1]
        data_sig_NoweightE = dataset["sig"][:, 0]
        data_bkg_NoweightE = dataset["bkg"][:, 0]
        data_sig_weightE = dataset["sig"][:, 1]
        data_bkg_weightE = dataset["bkg"][:, 1]
        data_sig_vertex = dataset["sig_vertex"]
        data_bkg_vertex = dataset["bkg_vertex"]
        data_sig_equen = dataset["sig_equen"]
        data_bkg_equen = dataset["bkg_equen"]

        if study_pdg:
            data_bkg_pdg = dataset["bkg_pdg"]
            data_bkg_px = dataset["bkg_px"]
            data_bkg_py = dataset["bkg_py"]
            data_bkg_pz = dataset["bkg_pz"]
        else:
            data_bkg_pdg, data_bkg_px, data_bkg_py, data_bkg_pz = [], [], [], []

        (data_sig_weightE, data_sig_NoweightE, data_bkg_weightE, data_bkg_NoweightE, data_sig_equen,
         data_bkg_equen, data_sig_vertex, data_bkg_vertex, data_bkg_pdg,
                data_bkg_px, data_bkg_py, data_bkg_pz ) = \
            self.VertexCut(data_sig_vertex, data_bkg_vertex, data_sig_weightE, data_bkg_weightE, data_sig_NoweightE,
                           data_bkg_NoweightE, data_sig_equen, data_bkg_equen, data_bkg_pdg, data_bkg_px,
                           data_bkg_py, data_bkg_pz, i_scheme)

        data_sig = np.concatenate((data_sig_NoweightE, data_sig_weightE), axis=1)
        data_bkg = np.concatenate((data_bkg_NoweightE, data_bkg_weightE), axis=1)

        # data_sig = data_sig_NoweightE
        # data_bkg = data_bkg_NoweightE

        # data_sig = np.concatenate((data_sig_NoweightE, data_sig_weightE, data_sig_vertex/10000, data_sig_equen.reshape((len(data_sig_equen), 1))),axis=1)
        # data_bkg = np.concatenate((data_bkg_NoweightE, data_bkg_weightE, data_bkg_vertex/10000, data_bkg_equen.reshape((len(data_bkg_equen), 1))),axis=1)

        print(f"len(data_sig):{len(data_sig)}, len(data_bkg):{len(data_bkg)}")

        n_validate = np.min( [len(data_bkg), len(data_sig)] )
        print(f"n_validate:{n_validate}")
        label_sig = np.ones(len(data_sig), dtype=np.int)
        label_bkg = np.zeros(len(data_bkg), dtype=np.int)
        self.input_validate = np.vstack((data_sig[:n_validate], data_bkg[:n_validate]))
        self.label_validate = np.concatenate((label_sig[:n_validate], label_bkg[:n_validate]))
        self.v_vertex_validate = np.concatenate((data_sig_vertex[:n_validate], data_bkg_vertex[:n_validate]))
        self.v_equen_validate = np.concatenate((data_sig_equen[:n_validate], data_bkg_equen[:n_validate]))

        if study_pdg:
            self.input_validate_bkg = data_bkg
            self.label_validate_bkg = label_bkg
            self.v_bkg_px, self.v_bkg_py, self.v_bkg_pz = data_bkg_px, data_bkg_py, data_bkg_pz
            self.v_bkg_pdg_validate = data_bkg_pdg

        print(f"shape of data :{self.input_validate.shape}")
        print(f" labels : {self.label_validate}")
        print(f"shape of vertices :{self.v_vertex_validate.shape}")
        print(f"shape of v_equqn  :{self.v_equen_validate.shape}")
        if len(self.input_validate) != len(self.label_validate):
            print("There existing a Error cause length of labels and data don't match !!!!!")
            exit(1)

    def LoadModelPredict(self, name_file_model):
        n_correct = 0
        n_correct_sig = 0
        n_correct_bkg = 0
        n_wrong_into_sig = 0
        n_wrong_into_bkg = 0

        # Load Model and do the predict
        with open(name_file_model, 'rb') as fr:
            new_svm = pickle.load(fr)

            for i in range(len(self.input_validate)):
                prediction = new_svm.predict(self.input_validate[i].reshape((1, len(self.input_validate[i]))))
                if prediction == self.label_validate[i]:
                    n_correct += 1
                    if self.label_validate[i] == 1:
                        n_correct_sig += 1
                    else:
                        n_correct_bkg += 1
                else:
                    if self.label_validate[i] == 1:
                        n_wrong_into_bkg += 1
                    else:
                        n_wrong_into_sig += 1
                print(f"Prediction : {prediction}, Anwser : {self.label_validate[i]}")
        print(f"Score : {n_correct / len(self.input_validate)} [{n_correct}/{len(self.input_validate)}]")
        print(
            f"correct_sig : {n_correct_sig / len(self.label_validate) * 2}   [{n_correct_sig}/{len(self.label_validate) / 2}]")
        print(
            f"correct_bkg : {n_correct_bkg / len(self.label_validate) * 2}   [{n_correct_bkg}/{len(self.label_validate) / 2}]")
        print(
            f"wrong into signal : {n_wrong_into_sig / len(self.label_validate) * 2} [{n_wrong_into_sig}/{len(self.label_validate) / 2}]")
        print(
            f"wrong into bkg    : {n_wrong_into_bkg / len(self.label_validate) * 2} [{n_wrong_into_bkg}/{len(self.label_validate) / 2}]")

    # def GetSigEff(self, v_eff_sig, v_eff_bkg, certain_eff_bkg=np.linspace(0, 0.05, 100)):
    def GetSigEff(self, v_eff_sig, v_eff_bkg, certain_eff_bkg=0.01):
        from scipy.interpolate import interp1d
        # eff_sig_return = griddata( v_eff_bkg, v_eff_sig, certain_eff_bkg , method="linear")
        v_eff_bkg = np.array(v_eff_bkg)
        v_eff_sig = np.array(v_eff_sig)
        f = interp1d(v_eff_bkg[1:], v_eff_sig[1:], kind="linear")
        eff_sig_return = f(certain_eff_bkg)
        # eff_sig_return = np.interp(certain_eff_bkg, v_eff_bkg, v_eff_sig)
        # print(f"bkg_eff:{v_eff_bkg}, sig_eff:{v_eff_sig}")
        # print(f"eff_sig_return : {eff_sig_return}")
        return (certain_eff_bkg, eff_sig_return)

    def LoadModelGetEfficiency(self, name_file_model, name_scheme=""):
        with open(name_file_model, 'rb') as fr:
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
        n0, bins0, patches0 = ax1.hist(tag_0, bins=np.linspace(0, 1, 600), color='red', histtype='step',
                                       label='Background')
        n1, bins1, patches1 = ax1.hist(tag_1, bins=np.linspace(0, 1, 600), color='blue', histtype='step',
                                       label='Signal')
        ax1.set_xlim(0, 1)
        plt.semilogy()
        ax1.legend()
        ax1.set_xlabel('Prediction output')
        fig1.savefig('predicts.png')

        eff_bkg = []
        eff_sig = []
        # print(f"n0: {n0}, \n n1: {n1}")
        for i in range(len(n0)):
            eff_bkg.append(np.sum(n0[i:]) * 1.0 / np.sum(n0))
            eff_sig.append(np.sum(n1[i:]) * 1.0 / np.sum(n1))
        # fig2, ax2 = plt.subplots()
        fig2 = plt.figure("Sig eff. VS Bkg eff.")
        ax2=fig2.add_subplot(111)
        ax2.plot(eff_bkg, eff_sig, label=name_scheme)
        (certain_eff_bkg, eff_sig_return) = self.GetSigEff(v_eff_bkg=eff_bkg, v_eff_sig=eff_sig)
        ax2.scatter(certain_eff_bkg, eff_sig_return, s=10, marker=(5, 1), label=name_scheme)
        print(f"background eff. : {certain_eff_bkg} ---> signal eff. : {eff_sig_return}")
        ax2.set_xlabel('Background efficiency')
        ax2.set_ylabel('Signal efficiency')
        ax2.set_xlim(0, 0.05)
        ax2.set_ylim(0, 1)
        plt.legend()
        fig2.savefig('roc.png')
        return float(eff_sig_return)
    def StudyEasyClassifybkg_R3andE(self, name_file_model, i_scheme=0 ):
        with open(name_file_model, 'rb') as fr:
            classifier = pickle.load(fr)
        predict_proba = classifier.predict_proba(self.input_validate)
        tag_0 = []
        tag_1 = []
        vertex_0 = []
        vertex_1 = []
        v_equen_0 = []
        v_equen_1 = []
        for i in range(len(predict_proba)):
            if self.label_validate[i] == 0:
                tag_0.append(predict_proba[i][1])
                vertex_0.append(self.v_vertex_validate[i])
                v_equen_0.append(self.v_equen_validate[i])
            else:
                tag_1.append(predict_proba[i][1])
                vertex_1.append(self.v_vertex_validate[i])
                v_equen_1.append(self.v_equen_validate[i])
        vertex_0, vertex_1 = np.array(vertex_0), np.array(vertex_1)
        X_0, Y_0, Z_0 = vertex_0[:, 0], vertex_0[:, 1], vertex_0[:, 2]
        X_1, Y_1, Z_1 = vertex_1[:, 0], vertex_1[:, 1], vertex_1[:, 2]
        R3_0 = np.sqrt(X_0**2+Y_0**2+Z_0**2)**3/1000.**3
        R3_1 = np.sqrt(X_1**2+Y_1**2+Z_1**2)**3/1000.**3
        v_equen_0, v_equen_1 = np.array(v_equen_0), np.array(v_equen_1)

        easy_criterion = 0.2
        hard_criterion = 0.5
        # if study_pdg:

        fig1 = plt.figure(f"Train scheme {i_scheme} ---R3")
        ax1 = fig1.add_subplot(111)
        plt.hist(R3_0[np.array(tag_0)<easy_criterion], bins=np.arange(0, 5000, 1000), histtype="step", label="$P_{signal}$"+f"<{easy_criterion}", density=True )
        plt.hist(R3_0[np.array(tag_0)>hard_criterion], bins=np.arange(0, 5000, 1000), histtype="step", label="$P_{signal}$"+f">{hard_criterion}", density=True )
        plt.hist(R3_0                                , bins=np.arange(0, 5000, 1000), histtype="step", label="Total",               density=True )
        plt.title("Distribution of Background Events $R^3$ ")
        plt.xlabel("$R^3$ [ $m^3$ ]")
        plt.ylabel("Probability")
        plt.legend()

        fig2 = plt.figure(f"Train scheme {i_scheme} --- Equen_bkg")
        bins_equen = np.arange(0, 70, 1)
        plt.hist(v_equen_0[np.array(tag_0)< easy_criterion], bins=bins_equen, histtype="step", label="$P_{signal}$"+f"<{easy_criterion}", density=True)
        plt.hist(v_equen_0[np.array(tag_0)> hard_criterion], bins=bins_equen, histtype="step", label="$P_{signal}$"+f">{hard_criterion}", density=True)
        plt.hist(v_equen_0                                 , bins=bins_equen, histtype="step", label="Total",               density=True)
        plt.title("Distribution of Background Events Equen ")
        plt.xlabel("Equen [ MeV ]")
        plt.ylabel("Probability")
        plt.legend()

        fig3 = plt.figure(f"Train scheme {i_scheme} --- Equen_sig")
        plt.hist(v_equen_1[np.array(tag_1)< easy_criterion], bins=bins_equen, histtype="step", label="$P_{signal}$"+f"<{easy_criterion}", density=True)
        plt.hist(v_equen_1[np.array(tag_1)> hard_criterion], bins=bins_equen, histtype="step", label="$P_{signal}$"+f">{hard_criterion}", density=True)
        plt.hist(v_equen_1                                 , bins=bins_equen, histtype="step", label="Total",               density=True)
        plt.title("Distribution of Signal Events Equen ")
        plt.xlabel("Equen [ MeV ]")
        plt.ylabel("Probability")
        plt.legend()
        plt.show()

    def StudyEasyClassifybkg_pdg(self, name_file_model, i_scheme=0 ):
        with open(name_file_model, 'rb') as fr:
            classifier = pickle.load(fr)

        predict_proba = classifier.predict_proba(self.input_validate_bkg)
        tag_0 = []
        for i in range(len(predict_proba)):
            tag_0.append(predict_proba[i][1])
        easy_criterion = 0.001
        hard_criterion = 0.5
        tag_0 = np.array(tag_0)
        index_easy = (np.array(tag_0)< easy_criterion)
        index_hard = (np.array(tag_0)> hard_criterion)
        pdg_easy = self.v_bkg_pdg_validate[index_easy]
        pdg_hard = self.v_bkg_pdg_validate[index_hard]

        generalize_dict_easy = {"nu_\mu":[], "nu_e":[], "p":[], "n":[], "N":[]}
        generalize_dict_hard = {"nu_\mu":[], "nu_e":[], "p":[], "n":[], "N":[]}

        def PutEvtIntoDict(generalize_dict:dict, event):
            evt_counter = Counter(event)
            generalize_dict["nu_e"].append(evt_counter[12]+evt_counter[-12])
            generalize_dict["nu_\mu"].append(evt_counter[14]+evt_counter[-14])
            generalize_dict["p"].append(evt_counter[2212])
            generalize_dict["n"].append(evt_counter[2112])
            generalize_dict["N"].append(0)
            for key in evt_counter.keys():
                if key > 1000000000:
                    generalize_dict["N"][-1] += evt_counter[key]
            return generalize_dict

        list_pdg_easy = []
        list_pdg_hard = []
        v_n_neutrons_oneevt_easy = []
        v_n_neutrons_oneevt_hard = []
        from collections import Counter
        for event in pdg_hard:
            list_pdg_hard.append(list(np.sort(event)))
            generalize_dict_hard = PutEvtIntoDict(generalize_dict_hard, event)
        for event in pdg_easy:
            list_pdg_easy.append(list(np.sort(event)))
            generalize_dict_easy  = PutEvtIntoDict(generalize_dict_easy, event)

        keys = list(generalize_dict_easy.keys())

        def ReformDict(dict_data:dict, keys):
            data_return = np.array(dict_data[keys[0]])[np.newaxis].T
            data_return_str = []
            for key in keys[1:]:
                data_return = np.concatenate((data_return, np.array(dict_data[key])[np.newaxis].T), axis=1)
            for arr in data_return:
                data_return_str.append("".join(list(np.array(arr, dtype=str))))
            return data_return_str

        generalize_array_easy = ReformDict(generalize_dict_easy, keys)
        generalize_array_hard = ReformDict(generalize_dict_hard, keys)
        generalize_counter_easy = Counter(generalize_array_easy)
        generalize_counter_hard = Counter(generalize_array_hard)
        print(f"Sig. Pro. < {easy_criterion} : ",generalize_counter_easy,"\n")
        print(f"Sig. Pro. > {hard_criterion} : ",generalize_counter_hard)
        df_easy = pd.DataFrame.from_dict(generalize_counter_easy, orient="index", columns=[f"Signal Probability < {easy_criterion}"])
        df_hard = pd.DataFrame.from_dict(generalize_counter_hard, orient="index", columns=[f"Signal Probability > {hard_criterion}"])
        df_plot = pd.concat([df_easy, df_hard], axis=1 )
        df_plot.sort_index(inplace=True, ascending=False)

        #### Normalize df_plot##########
        for i in list(df_plot.columns):
           # 获取各个指标的最大值和最小值
            Max = np.max(df_plot[i])
            df_plot[i] = df_plot[i]/Max

        df_plot.plot.barh()
        #### generate title ###########
        title_plot = f"${v_eff_condition[i_scheme].replace('$', '')}("
        for key in keys:
            if "nu" in key:
                title_plot += "\\"
            title_plot += key+":"
        title_plot = title_plot[:-1]+ "$)"
        ###############################
        plt.title(title_plot)
        plt.xlabel("Normalized Num. of Evts")
        plt.ylabel("Tag")
        # plt.show()

        #####################################################using 3D scatter to analyze pdg######################################################################################
        # keys = list(generalize_dict_easy.keys())
        # fig = plt.figure("visualize pdg")
        # ax = fig.add_subplot(111, projection="3d")
        # img1= ax.scatter(generalize_dict_easy[keys[0]], generalize_dict_easy[keys[1]], generalize_dict_easy[keys[2]], c=generalize_dict_easy[keys[3]], cmap=plt.hot(), s=3, marker=".")
        # img2= ax.scatter(generalize_dict_hard[keys[0]], generalize_dict_hard[keys[1]], generalize_dict_hard[keys[2]], c=generalize_dict_hard[keys[3]], cmap=plt.hot(), s=3, marker="^")
        # fig.colorbar(img1)
        # plt.show()
        #############################################################################################################################################################################

        # print(f"generalized evts dict(easy) : {generalize_dict_easy['n'][:10]}")
        # print(f"generalized evts dict(hard) : {generalize_dict_hard['n'][:10]}")
        # print(f"pdg_easy : {list_pdg_easy[:50]}")
        # print("\n")
        # print(f"pdg_hard : {list_pdg_hard[:20]}")

        #########################################plt the signal probability distribution of bkg############################################################################################
        # fig3 = plt.figure("easy VS hard")
        # ax3 = fig3.add_subplot(111)
        # n0, bins0, patches0 = ax3.hist(tag_0[index_easy], bins=np.linspace(0, 1, 600), color='red', histtype='step',
        #                                label=f'bkg(Sig. Pro. < {easy_criterion}')
        # n1, bins1, patches1 = ax3.hist(tag_0[index_hard], bins=np.linspace(0, 1, 600), color='blue', histtype='step',
        #                                label=f'bkg(Sig. Pro. > {hard_criterion}')
        # ax3.set_xlim(0, 1)
        # plt.semilogy()
        # ax3.legend()
        # ax3.set_xlabel('Prediction output')
        # plt.show()
        ###############################################################################################################################################################################

if __name__ == '__main__':
    import glob
    filelist_total = list(glob.glob("./jobs_DSNB_sk_data/data_withpdg/*.npz"))
    ratio_split = 0.1
    # filelist_train = filelist_total[:int((1-0.1)*len(filelist_total))]
    # filelist_validate = filelist_total[int((1-0.1)*len(filelist_total)):]
    filelist_train = filelist_total[:40]
    filelist_validate = filelist_total[40:]
    print(f"filelist_train : {filelist_train}")
    print(f"filelist_validate : {filelist_validate}")
    ana = SpectrumAna('FakeTest')
    v_eff_sig = {}
    filelist = ["test_fulltime_step10.npz", "test_fulltime_step5.npz"]
    v_eff_condition = {0:"$R^3$<4096", 1:"$R^3$<1000", 2:"1000<=$R^3$<2000", 3:"2000<=$R^3$<3000", 4:"3000<=$R^3$<4096"}
    study_pdg = False
    Equen_cut = True
    for i in range(5):
    # for i in [0]:
        print(f"#################processing {i} scheme##################")
        name_file_model = f"./model_maxtime_rebinning/model_maxtime_{i}.pkl"
        # # ana.loadDSNB("./try.npz")
        ana.loadDSNB(filelist_train, i_scheme=i)
        ana.AddModels()
        # ana.CompareModels()
        ana.TrainModel('MLPClassifier', pkl_filename=name_file_model)
        # ana.TrainModel('RandomForest', pkl_filename=name_file_model)

        # ana.LoadValidateData(filelist_validate, i_scheme=i)
        # # ana.StudyEasyClassifybkg_R3andE(name_file_model, i_scheme=i)
        # ana.StudyEasyClassifybkg_pdg(name_file_model, i_scheme=i)
    # ######################Validate model#################################################
    #     v_eff_sig[v_eff_condition[i]] = ana.LoadModelGetEfficiency(name_file_model, v_eff_condition[i])
    # print(f"Under Background eff. = 0.01, Signal eff.:{v_eff_sig}")
    # average_eff = 0
    # for key in v_eff_sig.keys():
    #     if key == '$R^3$<4096':
    #         continue
    #     average_eff += v_eff_sig[key]
    # average_eff /= (len(v_eff_sig.keys())-1)
    # print(f"Average Signal eff. : {average_eff}")
    # plt.show()

