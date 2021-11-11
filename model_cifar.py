import torch
import torch.nn as nn
import torch.nn.functional as F


class ClassifierNet(nn.Module):
    def __init__(self):
        super(ClassifierNet, self).__init__()

        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):

        activations = [x] # conv1, conv2, linear1, linear2
        x = F.relu(self.conv1(x))
        activations.append(x)
        x = self.pool(x)

        x = F.relu(self.conv2(x))
        activations.append(x)
        x = self.pool(x)

        x = torch.flatten(x, 1) # flatten all dimensions except batch
        x = F.relu(self.fc1(x))
        activations.append(x)

        x = F.relu(self.fc2(x))
        activations.append(x)

        x = self.fc3(x)
        output = F.log_softmax(x, dim=1)
        activations.append(output)
        return output, activations


class DetectorNet(nn.Module):
    def __init__(self):
        super(DetectorNet, self).__init__()
        
        self.simple_linear0 = nn.Linear( 1, 1)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv1 = nn.Conv2d(3, 6, 3)
        self.conv1_bn = nn.BatchNorm2d(6)

        self.conv2 = nn.Conv2d(6, 16, 3)
        self.conv2_bn = nn.BatchNorm2d(16)
        self.conv3 = nn.Conv2d(16, 1, 3)
        self.conv3_bn = nn.BatchNorm2d(1)
        self.conv4 = nn.Conv2d(1, 1, 3)


    def forward(self, activations):
        input, conv1, conv2, linear1, linear2 = activations[0], activations[1], activations[2], activations[3], activations[4]
        x = self.conv1(input)
        conv0_out = F.relu(x)
        x = self.pool(conv1)

        x = self.conv2(x)
        conv1_out = F.relu(x)
        x = self.pool(conv2)

        x = F.relu(self.conv3_bn(self.conv3(x)))  

        x = F.relu(self.conv4(x)) 

        x = torch.flatten(x, 1)
        
        x = F.relu(self.simple_linear0(x))

        output = x
        return output

      
        '''
        x = torch.flatten(input, 1)
        x = self.simple_linear0(x)
        x = F.relu(x)
        x = self.simple_linear1(x)
        output = x
      

        x = self.conv0(input)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = self.conv0l(x)
        conv0_out = F.relu(x)
        
        x = self.conv1(conv1)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        
        x = torch.flatten(x, 1)
        x = self.conv1l(x)
        conv1_out = F.relu(x)
        
        x = self.conv2(conv2)
        x = F.relu(x)
        x = torch.flatten(x, 1)
        x = self.conv2l(x)
        conv2_out = F.relu(x)
        
        x = self.linear1(linear1)
        linear1_out = F.relu(x)
        
        x = self.linear2(linear2)
        linear2_out = F.relu(x)
        
        x = torch.cat((conv0_out, conv1_out, conv2_out, linear1_out, linear2_out), 1)
        x = conv0_out
        x = self.fc1(x)
        x = F.relu(x)
        output = self.fc2(x)
        '''


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.classifier = ClassifierNet()
        self.detector = DetectorNet()

    def forward(self, x):
        c_output, activations = self.classifier.forward(x)
        d_output = self.detector.forward(activations)
        return c_output, d_output

    def parameters(self, which="all"):
        if which == "detector":
            return self.detector.parameters()
        elif which == "classifier":
            return self.classifier.parameters()
        else:
            return super().parameters()

