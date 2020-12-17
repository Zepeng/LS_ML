import numpy as np
import matplotlib.pyplot as plt

results = np.load('test_score_1.npy', allow_pickle=True)[3]
signal = []
background = []
for result in results:
    if result[2] < 11 or result[2] > 30:
        continue
    if result[1] == 0:
        background.append(result[0][1])
    else:
        signal.append(result[0][1])

n, bins, patches = plt.hist(background, bins=np.linspace(0.0, 1, 1001), histtype='step', color='blue', label='Background')
print(n*100.0/np.sum(n))
n, bins, patches = plt.hist(signal, bins=np.linspace(0.0, 1, 1001), histtype='step', color='red', label='Signal')
print(n*100.0/np.sum(n))
plt.legend()
plt.xlim(0.0,1)
plt.xlabel('DNN output')
plt.yscale('log')
plt.savefig('DSNB_classification.pdf')

