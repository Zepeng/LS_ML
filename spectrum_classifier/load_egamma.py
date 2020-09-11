import uproot as up
import numpy as np
import matplotlib.pyplot as plt

def LoadEGamma(rootfile, isTest=False):
    tree = up.open(rootfile)['waves']
    wave_sum = tree.array('waves_Sum')
    gamma_tag = tree.array('isGamma')
    time_sequence = np.arange(len(wave_sum[0]))
    events = []
    for evtid in range( len(wave_sum) ):
        wave_instance = wave_sum[evtid]
        bins, edges = np.histogram(time_sequence, bins = 10,  weights = wave_instance)
        events.append([bins, gamma_tag[evtid]])
    return np.array(events)

def test():
    events = LoadEGamma('/hpcfs/juno/junogpu/luoxj/Data_PSD/elecsim_Electron_001_SumAllPmtWaves.root')
    print(events)

#test()
