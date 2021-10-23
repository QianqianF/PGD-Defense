import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        # self.dropout1 = nn.Dropout(0.25)
        # self.dropout2 = nn.Dropout(0.5)
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        activations = [] # conv1, conv2, linear1, linear2

        x = self.conv1(x)
        conv1 = F.relu(x) # n, 32, 
        activations.append(conv1)

        x = self.conv2(conv1)
        x = F.relu(x)
        conv2 = F.max_pool2d(x, 2)
        activations.append(conv2)

        # x = self.dropout1(x)
        x = torch.flatten(conv2, 1)
        x = self.fc1(x)
        linear1 = F.relu(x)
        activations.append(linear1)

        # x = self.dropout2(x)
        x = self.fc2(linear1)
        output = F.log_softmax(x, dim=1)
        activations.append(output)

        return output, activations


class DetectorNet(nn.Module):
    def __init__(self):
        super(DetectorNet, self).__init__()
        self.conv1 = nn.Conv2d(32, 64, 3, 1)
        self.conv1l = nn.Linear(9216, 64)

        self.conv2 = nn.Conv2d(64, 64, 3, 1)
        self.conv2l = nn.Linear(9216, 64)

        self.linear1 = nn.Linear(128, 64)
        self.linear2 = nn.Linear(10, 1)

        self.fc1 = nn.Linear(193, 10)
        self.fc2 = nn.Linear(10, 1)

    def forward(self, activations):
        conv1, conv2, linear1, linear2 = activations[0], activations[1], activations[2], activations[3]
        x = self.conv1(conv1)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.conv1l(x)
        conv1_out = F.relu(x) # n, 64

        x = self.conv2(conv2)
        conv2_out = F.relu(x) # 

        x = self.linear1(linear1)
        linear1_out = F.relu(x)

        x = self.linear2(linear2)
        linear2_out = F.relu(x)

        x = torch.cat((conv1_out, conv2_out, linear1_out, linear2_out), 1)
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)

        return output

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = ClassifierNet()
        self.detector = DetectorNet()

    def forward(self, x):
        c_output, activations = self.classifier.forward()
        d_output = self.detector.forward()

    


