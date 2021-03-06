import torch
from torch.nn import Module
from copy import deepcopy
from torch.nn.functional import softmax
from torch.distributions.multivariate_normal import MultivariateNormal

class SWAGDiagonalModel(Module):
    def __init__(self, model, device=None, mode='parameters'):
        super(SWAGDiagonalModel, self).__init__()
        self.mean = deepcopy(model)
        self.second_moment = deepcopy(model)
        self.inference_model = None
        if device is not None:
            self.mean = self.mean.to(device)
            self.second_moment = self.second_moment.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        def avg_fn(mean, second_moment, model_parameter, num_averaged):
            return (mean + (model_parameter - mean) / (num_averaged + 1),
                second_moment + (model_parameter**2 - second_moment) / (num_averaged + 1))
        self.avg_fn = avg_fn
        modes = ['parameters', 'state_dict']
        if mode not in modes:
            raise ValueError(f'Invalid mode passed, valid values are {", ".join(modes)}.')
        self.use_state_dict = mode == 'state_dict'

    def forward(self, n_samples, *args, **kwargs):
        if self.inference_model is None:
            self.inference_model = deepcopy(self.mean).to('cuda')
        swa_mean_param = self.mean.state_dict().values() if self.use_state_dict else self.mean.parameters()
        swa_second_moment_param = self.second_moment.state_dict().values() if self.use_state_dict else self.second_moment.parameters()
        model_param = self.inference_model.state_dict().values() if self.use_state_dict else self.inference_model.parameters()
        class_probabilities_sum = 0. 
        self.train() # update BN statistics during inference
        for _ in range(n_samples):
            for p_swa_mean, p_swa_second_moment, p_model in zip(swa_mean_param, swa_second_moment_param, model_param):
                p_model.detach().copy_(torch.randn(p_model.shape).cuda() * torch.sqrt(p_swa_second_moment - p_swa_mean**2) + p_swa_mean)
            class_probabilities_sum += softmax(self.inference_model(*args, **kwargs), dim=1)
        self.eval()
        return class_probabilities_sum / n_samples

    def update_parameters(self, model):
        swa_mean_param = self.mean.state_dict().values() if self.use_state_dict else self.mean.parameters()
        swa_second_moment_param = self.second_moment.state_dict().values() if self.use_state_dict else self.second_moment.parameters()
        model_param = model.state_dict().values() if self.use_state_dict else model.parameters()
        for p_swa_mean, p_swa_second_moment, p_model in zip(swa_mean_param, swa_second_moment_param, model_param):
            device = p_swa_mean.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa_mean.detach().copy_(p_model_)
                p_swa_second_moment.detach().copy_(p_model_**2)
            else:
                mean, second_moment = self.avg_fn(p_swa_mean.detach(), p_swa_second_moment.detach(),  p_model_,
                                                 self.n_averaged.to(device))
                p_swa_mean.detach().copy_(mean)
                p_swa_second_moment.detach().copy_(second_moment)
        self.n_averaged += 1

class SWAGModel(Module):
    def __init__(self, model, k, device=None, mode='parameters'):
        super(SWAGModel, self).__init__()
        self.mean = deepcopy(model)
        self.second_moment = deepcopy(model)
        self.inference_model = None
        assert(k >= 1)
        self.k = k
        self.columns = []
        for _ in range(k):
            self.columns.append(deepcopy(model))
        self.filled_columns = 0
        if device is not None:
            self.mean = self.mean.to(device)
            self.second_moment = self.second_moment.to(device)
        self.register_buffer('n_averaged',
                             torch.tensor(0, dtype=torch.long, device=device))
        def avg_fn(mean, second_moment, model_parameter, num_averaged):
            return (mean + (model_parameter - mean) / (num_averaged + 1),
                second_moment + (model_parameter**2 - second_moment) / (num_averaged + 1))
        self.avg_fn = avg_fn
        modes = ['parameters', 'state_dict']
        if mode not in modes:
            raise ValueError(f'Invalid mode passed, valid values are {", ".join(modes)}.')
        self.use_state_dict = mode == 'state_dict'

    def forward(self, n_samples, *args, **kwargs):
        if self.inference_model is None:
            self.inference_model = deepcopy(self.mean).to('cuda')
        swa_mean_param = self.mean.state_dict().values() if self.use_state_dict else self.mean.parameters()
        swa_second_moment_param = self.second_moment.state_dict().values() if self.use_state_dict else self.second_moment.parameters()
        swa_columns = []
        for column in self.columns:
            swa_columns.append(column.state_dict().values() if self.use_state_dict else column.parameters())
        model_param = self.inference_model.state_dict().values() if self.use_state_dict else self.inference_model.parameters()
        class_probabilities_sum = 0. 
        self.train() # update BN statistics during inference
        for _ in range(n_samples):
            for p_swa_mean, p_swa_second_moment, p_model, *p_columns in zip(swa_mean_param, swa_second_moment_param, model_param, *swa_columns):
                stack = torch.stack(p_columns)
                # print(p_model.shape, stack.shape, (self.k, len()))
                p_model.detach().copy_(p_swa_mean + 2**-0.5 * torch.sqrt(p_swa_second_moment - p_swa_mean**2) * torch.randn(p_model.shape, device='cuda')
                + (2 * (self.k - 1))**-0.5 * torch.matmul(stack.T, torch.randn((self.k,), device='cuda')).T) # torch.randn((self.k, *((len(stack.shape) - 1) * (1,))), device='cuda'))
            class_probabilities_sum += softmax(self.inference_model(*args, **kwargs), dim=1)
        self.eval()
        return class_probabilities_sum / n_samples

    def update_parameters(self, model):
        swa_mean_param = self.mean.state_dict().values() if self.use_state_dict else self.mean.parameters()
        swa_second_moment_param = self.second_moment.state_dict().values() if self.use_state_dict else self.second_moment.parameters()
        swa_columns = []
        for column in self.columns:
            swa_columns.append(column.state_dict().values() if self.use_state_dict else column.parameters())
        model_param = model.state_dict().values() if self.use_state_dict else model.parameters()
        for p_swa_mean, p_swa_second_moment, p_model, *p_columns in zip(swa_mean_param, swa_second_moment_param, model_param, *swa_columns):
            device = p_swa_mean.device
            p_model_ = p_model.detach().to(device)
            if self.n_averaged == 0:
                p_swa_mean.detach().copy_(p_model_)
                p_swa_second_moment.detach().copy_(p_model_**2)
            else:
                mean, second_moment = self.avg_fn(p_swa_mean.detach(), p_swa_second_moment.detach(),  p_model_,
                                                 self.n_averaged.to(device))
                p_swa_mean.detach().copy_(mean)
                p_swa_second_moment.detach().copy_(second_moment)
            if self.filled_columns < self.k:
                p_columns[self.filled_columns].detach().copy_(p_model_ - p_swa_mean.detach())
                self.filled_columns += 1
            else:
                p_columns.append(p_columns.pop(0))
                p_columns[-1].detach().copy_(p_model_ - p_swa_mean.detach())

        self.n_averaged += 1