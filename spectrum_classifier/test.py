from torch import nn
import torch

class CNN1D(nn.Module):
    def __init__(self):
        super(CNN1D, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv1d(1, 128, kernel_size=7, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=7, stride=3),
            nn.Dropout(0.1),
        )

        self.conv2 = nn.Sequential(
            nn.Conv1d(128, 128, kernel_size=3, stride=3),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=3, stride=3),
            nn.Dropout(0.25)
        )

        self.fc1 = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Dropout(0.5)
        )

        self.fc2 = nn.Sequential(
            nn.Linear(512, 11),
            nn.ReLU(),
            nn.Dropout(0.5)
        )
        self.fc3 = nn.Linear(11, 2)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        # collapse
        x = x.view(x.size(0), -1)
        print(x.shape)
        # linear layer
        x = self.fc1(x)
        # output layer
        x = self.fc2(x)
        x = self.fc3(x)

        return x

def test():
    model = CNN1D()

    x = torch.randn(64, 1, 200)
    out = model(x)
    print(out.shape)
    #torch.Size([64, 11])
