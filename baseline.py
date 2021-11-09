from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt
import numpy as np

from model import Net, ClassifierNet
from pgd import pgd_
from main import apply_pgd, train_classifier, test_pgd_perturbed

def adversarial_train_classifier(args, model, device, train_loader, optimizer_classifier, epoch):
	model.train()
	for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
		x_batch, y_batch = x_batch.to(device), y_batch.to(device)
		x_batch, y_batch = apply_pgd(x_batch, y_batch, model, args.pgd_samples, args.max_pgd_delta, device, test_mode=False)

		optimizer.zero_grad()
		output, _ = model(x_batch)
		loss = F.nll_loss(output, y_batch)
		loss.backward()
		optimizer.step()
        
		if batch_idx % args.log_interval == 0:
		    print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
		        epoch, batch_idx * len(x_batch), len(train_loader.dataset),
		        100. * batch_idx / len(train_loader), loss.item()))
		    if args.dry_run:
		        break

def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=4, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=4, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 5)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=False,
                        help='For Loading the last Model')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Whether data should be perturbed when training the classifier (default: False)')
    parser.add_argument('--pgd-samples', type=int, default=7,
                        help='number of pgd samples used in adversarial training')
    parser.add_argument('--max-pgd-delta', type=float, default=0.1,
                        help='Max perturbation for pgd step')

    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform=transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize((0.1307,), (0.3081,))
        ])
    dataset1 = datasets.MNIST('../data', train=True, download=True,
                       transform=transform)
    dataset2 = datasets.MNIST('../data', train=False,
                       transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1,**train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer_classifier = optim.Adadelta(model.parameters("classifier"), lr=args.lr)
    optimizer_detector = optim.Adadelta(model.parameters("detector"), lr=0.1)

    # regularly train the model
    if args.load_model:
        model.load_state_dict(torch.load("mnist_cnn_std.pt", map_location=device))
    else:
        scheduler = StepLR(optimizer_classifier, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train_classifier(args, model, device, train_loader, optimizer_classifier, epoch)
            test(model, device, test_loader, args)
            scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn_std.pt")

    # continue training with PGD examples
    adversarial_train_classifier(args, model, device, train_loader, optimizer_classifier, epoch)

    test_pgd_perturbed(model, device, test_loader)

if __name__ == '__main__':
    main()