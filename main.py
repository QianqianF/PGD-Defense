from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau
import matplotlib.pyplot as plt

from model import Net, ClassifierNet


def train_classifier(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output, _ = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def fgsm_(model_, device, x, target, eps, targeted=True, clip_min=None, clip_max=None):
    """Internal process for all FGSM and PGD attacks."""
    # create a copy of the input, remove all previous associations to the compute graph...
    input_ = x.clone().detach_()
    # ... and make sure we are differentiating toward that variable
    input_.requires_grad_()

    # run the model and obtain the loss
    logits, _ = model_(input_)
    target = torch.LongTensor([target]).to(device)
    model_.zero_grad()
    loss = nn.CrossEntropyLoss()(logits, target)
    loss.backward()

    # perfrom either targeted or untargeted attack
    if targeted:
        out = input_ - eps * input_.grad.sign()
    else:
        out = input_ + eps * input_.grad.sign()

    # if desired clip the ouput back to the image domain
    if (clip_min is not None) or (clip_max is not None):
        out.clamp_(min=clip_min, max=clip_max)
    return out


def pgd_(model_, device, x, target, k, eps, eps_step, targeted=True, clip_min=None, clip_max=None):
    x = x.clone().detach()

    x_min = x - eps
    x_max = x + eps

    # Randomize the starting point x.
    # x = x + eps * (2 * torch.rand_like(x) - 1)
    # if (clip_min is not None) or (clip_max is not None):
    #     x.clamp_(min=clip_min, max=clip_max)

    xs = [x.clone().detach()]
    for i in range(k):
        # FGSM step
        # We don't clamp here (arguments clip_min=None, clip_max=None)
        # as we want to apply the attack as defined
        x = fgsm_(model_, device, x, target, eps_step, targeted)
        # Projection Step
        x = torch.max(x_min, x)
        x = torch.min(x_max, x)
        xs.append(x.clone().detach())
    # if desired clip the output back to the image domain
    xs = torch.stack(xs)
    if (clip_min is not None) or (clip_max is not None):
        xs.clamp_(min=clip_min, max=clip_max)
    return xs[:, 0, :, :, :]


def train_detector(model, device, train_loader, optimizer, epoch, args):
    model.train()
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        steps = 1
        x_perturbed = torch.zeros(steps + 1, *x_batch.size()).to(device)
        for i in range(len(x_batch)):
            pgd_images = pgd_(model, device, x_batch[None, i, :, :, :], y_batch[None, i], steps, 0.1, (0.1 / steps), targeted=False, clip_min=0., clip_max=1.)
            x_perturbed[:, i, :, :, :] = pgd_images
        # print(x_perturbed)
        flattened = x_perturbed.reshape(-1, *x_perturbed.size()[2:])
        labels = torch.tensor([])
        list = []
        for i in range(steps + 1):
            list.append(torch.ones(x_batch.shape[0]).to(device) * i)
        labels = torch.cat(list)

        # output, _ = model(flattened)
        # pred = output.argmax(dim=1, keepdim=True)
        # plt.imshow(flattened[-1, ...].permute(1, 2, 0).cpu(), cmap='gray')
        # plt.show()
        # print(42)

        perm = torch.randperm(len(labels))
        data, target = flattened[perm, ...], labels[perm]
        optimizer.zero_grad()
        _, output = model(data)
        loss = nn.MSELoss()(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Detector Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset) * (steps+1),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break

def save_ad_examples(model, device, train_loader):

    train_size = len(train_loader.dataset)
    steps = 7
    num_channel = 1
    im_size = 28

    ad_examples = torch.zeros([train_size * (steps + 1), num_channel, im_size, im_size])
    ad_labels = torch.zeros(train_size * (steps + 1))

    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):

        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_perturbed = torch.zeros(steps + 1, *x_batch.size()).to(device)
        for i in range(len(x_batch)):
            pgd_images = pgd_(model, device, x_batch[None, i, :, :, :], y_batch[None, i], steps, 0.1, 2.5 * (0.1 / steps), targeted=False, clip_min=0., clip_max=1.)
            x_perturbed[:, i, :, :, :] = pgd_images

        flattened = x_perturbed.reshape(-1, *x_perturbed.size()[2:])
        labels = torch.tensor([])
        list = []
        for i in range(steps + 1):
            list.append(torch.ones(x_batch.shape[0]).to(device) * i)
        labels = torch.cat(list)

        ad_examples[batch_idx * (train_loader.batch_size) * (steps+1):(batch_idx+1) * (train_loader.batch_size) * (steps+1)] = flattened
        ad_labels[batch_idx * (train_loader.batch_size) * (steps+1):(batch_idx+1) * (train_loader.batch_size) * (steps+1)] = labels

        print('Process PGD: [{}/{}]'.format(
                batch_idx * len(x_batch), train_size))

    torch.save(ad_examples, 'ad_examples.pt')
    torch.save(ad_labels, 'ad_labels.pt')




def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=5, metavar='N',
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
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=True,
                        help='For Loading the last Model')
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

    if args.load_model:
        model.load_state_dict(torch.load("mnist_cnn.pt"))
    else:
        scheduler = StepLR(optimizer_classifier, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train_classifier(args, model, device, train_loader, optimizer_classifier, epoch)
            test(model, device, test_loader)
            scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn.pt")


    # save_ad_examples(model, device, train_loader)

    scheduler = ReduceLROnPlateau(optimizer_detector, patience=2)
    for epoch in range(1, args.epochs + 1):
        train_detector(model, device, train_loader, optimizer_detector, epoch, args)


if __name__ == '__main__':
    main()