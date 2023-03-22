from torch import nn

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.bn1 = nn.BatchNorm1d(1024)
        self.fc1 = nn.Linear(1024, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc3 = nn.Linear(128, 10)
        # self.fc4 = nn.Dropout(p=0.2)

    def forward(self, x):
        x = self.bn1(x)
        x = self.fc1(x)
        # x = self.fc4(x)
        x = nn.functional.relu(x)
        x = self.bn2(x)
        x = nn.functional.relu(x)
        x = self.fc2(x)
        x = self.bn3(x)
        x = nn.functional.relu(x)
        x = self.fc3(x)
        return nn.functional.log_softmax(x, dim=-1)
