import numpy
import torch

class Simple1DCNN(torch.nn.Module):
    def __init__(self):
        super(Simple1DCNN, self).__init__()
        self.layer1 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=5, stride=2)
        self.act1 = torch.nn.ReLU()
        self.layer2 = torch.nn.Conv1d(in_channels=1, out_channels=1, kernel_size=1)
        self.fc = torch.nn.Linear(99, 2)
    def forward(self, x):
        x = self.layer1(x)
        x = self.act1(x)
        x = self.layer2(x)
        x = self.fc(x)

        return x
def test():
    X = numpy.random.uniform(-10, 10, 402).reshape(2, 1, -1)
    print(X, X.shape, type(X))
    model = Simple1DCNN().double()
    print(model(torch.tensor(X)).shape)
