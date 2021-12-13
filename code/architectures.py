from archs.cifar_resnet import resnet as resnet_cifar
from datasets import get_normalize_layer, get_input_center_layer
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
from torch.nn.functional import interpolate
from torchvision.models.resnet import resnet50
from blitz.models.b_vgg import vgg11, VGG, make_layers, cfg
from blitz.utils import variational_estimator
from blitz.modules import BayesianLinear, BayesianConv2d
import math

from swag import SWAG
# resnet50 - the classic ResNet-50, sized for ImageNet
# cifar_resnet20 - a 20-layer residual network sized for CIFAR
# cifar_resnet110 - a 110-layer residual network sized for CIFAR
ARCHITECTURES = ["cifar_resnet20", "resnet50", "cifar_resnet110", "imagenet32_resnet110", "cifar_bbb_vgg11"]

@variational_estimator
class NormalizedVgg(nn.Module):
    '''
    VGG model 
    '''
    def __init__(self, dataset, features, out_nodes=10):
        super(NormalizedVgg, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            #nn.Dropout(),
            BayesianLinear(512, 512),
            nn.ReLU(True),
            #nn.Dropout(),
            BayesianLinear(512, 512),
            nn.ReLU(True),
            BayesianLinear(512, out_nodes),
        )
        
        for m in self.modules():
            if isinstance(m, BayesianConv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight_mu.data.normal_(0, math.sqrt(2. / n))
                m.bias_mu.data.zero_()

        self.normalize_layer = get_normalize_layer(dataset)
    
    def forward(self, x):
        x = self.normalize_layer(x)
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

    def sample_elbo_with_output(self, inputs, labels,
                criterion, sample_nbr, complexity_cost_weight=1):
        loss = 0.
        outputs = 0.
        for _ in range(sample_nbr):
            output = self(inputs)
            outputs += output
            loss += criterion(output, labels) 
            loss += self.nn_kl_divergence() * complexity_cost_weight
        return loss / sample_nbr, outputs / sample_nbr

def VGG_SWAG(nn.Module):
    def __init__(self, dataset, features, out_nodes=10, no_cov_mat=True):
        super(VGG_SWAG, self).__init__()
        self.model = SWAG(NormalizedVgg, no_cov_mat=no_cov_mat, dataset, features)
        
    def forward(self, x):
        return self.model.forward(x)

    def sample(self, inputs, labels, criterion, samples_nbr, complexity_cost_weight):




def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """ Return a neural network (with random weights)

    :param arch: the architecture - should be in the ARCHITECTURES list above
    :param dataset: the dataset - should be in the datasets.DATASETS list
    :return: a Pytorch module
    """
    if arch == "resnet50" and dataset == "imagenet":
        model = torch.nn.DataParallel(resnet50(pretrained=False)).cuda()
        cudnn.benchmark = True
    elif arch == "cifar_resnet20":
        model = resnet_cifar(depth=20, num_classes=10).cuda()
    elif arch == "cifar_resnet110":
        model = resnet_cifar(depth=110, num_classes=10).cuda()
    elif arch == "imagenet32_resnet110":
        model = resnet_cifar(depth=110, num_classes=1000).cuda()
    elif arch == "cifar_bbb_vgg11":
        return NormalizedVgg(dataset, make_layers(cfg['A'])).cuda()

    # Both layers work fine, We tried both, and they both
    # give very similar results 
    # IF YOU USE ONE OF THESE FOR TRAINING, MAKE SURE
    # TO USE THE SAME WHEN CERTIFYING.
    normalize_layer = get_normalize_layer(dataset)
    # normalize_layer = get_input_center_layer(dataset)
    return torch.nn.Sequential(normalize_layer, model)
