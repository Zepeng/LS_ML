import uproot as up
import numpy as np
import matplotlib.pyplot as plt

def LoadCalib(rootfile, eventtype):
    if eventtype not in ['FastNeutron', 'CoTree']:
        print('Wrong Event Type')
        return 0
    if eventtype == 'FastNeutron':
        tree = up.open(rootfile)['JRHu/tree/%s' % eventtype]
        promptcharge = tree.array('pmtPromptCharge')
        prompttime = tree.array('pmtPromptTime')
        print(np.max(prompttime), np.min(prompttime))
        events = []
        for evtid in range( len(prompttime) ):
            eventcharge = promptcharge[evtid]
            eventtime = prompttime[evtid]
            qevt = []
            tevt = []
            for i in range(24):
                for j in range(8):
                    for k in range(5):
                        if eventcharge[i][j][k] > 0:
                            qevt.append(eventcharge[i][j][k])
                            tevt.append(eventtime[i][j][k])
            bins, edges = np.histogram(tevt, bins = 10, range=(-1650, -850), weights = qevt)
            events.append(bins/np.sum(bins))
        np.save('%s.npy' % eventtype, np.array(events))
        return np.array(events)
    else:
        tree = up.open(rootfile)['JRHu/tree/%s' % eventtype]
        promptcharge = tree.array('pmtCharge')
        prompttime = tree.array('pmtTime')
        print(np.max(prompttime), np.min(prompttime))
        events = []
        for evtid in range(len(prompttime)):
            eventcharge = promptcharge[evtid]
            eventtime = prompttime[evtid]
            qevt = []
            tevt = []
            for i in range(24):
                for j in range(8):
                    for k in range(5):
                        if eventcharge[i][j][k] > 0:
                            qevt.append(eventcharge[i][j][k])
                            tevt.append(eventtime[i][j][k])
            bins, edges = np.histogram(tevt, bins = 10, range=(-1650, -850), weights = qevt)
            events.append(bins/np.sum(bins))
        np.save('%s.npy' % eventtype, np.array(events))
        return np.array(events)

def test():
    events = LoadCalib('/junofs/users/lirh/DYB/run67527.root', 'FastNeutron')
    print(events.shape)
    events = LoadCalib('/junofs/users/lirh/DYB/run67522.root', 'CoTree')
    print(events.shape)

#test()
