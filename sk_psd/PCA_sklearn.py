from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import numpy as np
class PCA_tool:
    def __init__(self, rawinput_0, rawinput_1):
        self.rawinput_0 = rawinput_0
        self.rawinput_1 = rawinput_1
        self.datainput = np.array([])
    def PrepareDataFormat_DSNB(self):
        self.datainput = np.array(self.rawinput_0[list(self.rawinput_0.keys())[0]])[np.newaxis].T
        for key in list(self.rawinput_0.keys())[1:]:
            self.datainput = np.concatenate((self.datainput,np.array(self.rawinput_0[key])[np.newaxis].T), axis=1)




