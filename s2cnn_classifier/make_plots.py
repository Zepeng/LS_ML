import numpy as np
import matplotlib.pyplot as plt

result_dir = "./usgcnn_train_TimeNearest/"
result_dir = "./usgcnn_train_TimeNearest_lr0.5decay/"
results = np.load(result_dir+'test_score_1.npy', allow_pickle=True)[3]
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

result_loss = np.load(result_dir+'loss_acc.npy', allow_pickle=True)
result_loss = result_loss[:, :20]
print(result_loss)
train_loss, train_acc, test_loss, test_acc = (result_loss[0], result_loss[1], result_loss[2], result_loss[3])

plt.figure("Accuracy")
plt.plot(train_acc, label= "Train")
plt.plot(test_acc  , label= "Test")
plt.legend()
plt.xlabel("epoch")
plt.title("Accuracy")

plt.figure("Loss")
plt.plot(train_loss, label= "Train")
plt.plot(test_loss  , label= "Test")
plt.legend()
plt.xlabel("epoch")
plt.title("Loss")

plt.show()
