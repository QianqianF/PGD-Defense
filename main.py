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


def train_classifier(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        if args.augment:
            data += (torch.rand(data.shape, device=device) * 2. - 1.) * 0.2
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


# returns (# of pgd_samples) pgd perturbed images with max_pgd_delta
# x_batch_pgd: ((pgd_samples + 1) * x_batch.size, color_channel, image_width, image_height)
def apply_pgd(x_batch, y_batch, model, pgd_samples, max_pgd_delta, device, test_mode=False):

    y_batch_pgd = torch.tensor(0., device=device)
    if pgd_samples > 0:
        steps = pgd_samples
        x_perturbed = torch.zeros(steps + 1, *x_batch.size()).to(device)
        for i in range(len(x_batch)):
            pgd_images = pgd_(model, device, x_batch[None, i, :, :, :], y_batch[None, i], steps, max_pgd_delta, (max_pgd_delta / steps),
                              targeted=False, clip_min=0., clip_max=1., random_start=test_mode)
            x_perturbed[:, i, :, :, :] = pgd_images

        if test_mode:
            x_batch_pgd = x_perturbed[-1, ...]
        else:
            x_batch_pgd = x_perturbed.reshape(-1, *x_perturbed.size()[2:])
            list = []
            for i in range(steps + 1):
                list.append(torch.ones(x_batch.shape[0]).to(device) * i)
            y_batch_pgd = torch.true_divide(torch.cat(list), pgd_samples)

        return x_batch_pgd, y_batch_pgd
    return x_batch, torch.tensor(0.)

# max_perturbation: max_uniform_delta if noise="uniform"; max_sigma if noise="gaussian"
def apply_noise(x_batch, model, noise_samples, max_perturbation, device, noise="gaussian", test_mode=False):

    y_batch_noise = torch.tensor(0., device=device)
    if noise_samples > 0:
        if test_mode:
            ceils = torch.tensor([max_perturbation]).to(device)
        else:
            ceils = torch.arange(0., max_perturbation + torch.finfo(torch.float32).eps, max_perturbation / noise_samples).to(device)

        if noise == "gaussian":
            x_batch_noise_temp = x_batch + torch.randn((noise_samples + 1, *x_batch.shape), device=device) * ceils[:, None, None, None, None]
        elif noise == "uniform":
            x_batch_noise_temp = x_batch + (torch.rand((noise_samples + 1, *x_batch.shape), device=device) * 2. - 1.) * ceils[:, None, None, None, None]
        else:
            raise NotImplementedError()

        x_batch_noise = torch.flatten(x_batch_noise_temp, start_dim=0, end_dim=1)
        y_batch_noise = torch.repeat_interleave(torch.true_divide(ceils, max_perturbation), len(x_batch))

    return x_batch_noise, y_batch_noise


# apply noise to both clean and pgd perturbed images
# max_perturbation: max_uniform_delta if noise="uniform"; max_sigma if noise="gaussian"
def sample_perturbed_data(x_batch, y_batch, model, pgd_samples, max_pgd_delta, noise_samples, max_perturbation, device, noise="gaussian", test_mode=False):

    x_batch_pgd, y_batch_pgd = apply_pgd(x_batch, y_batch, model, pgd_samples, max_pgd_delta, device, test_mode)

    x_batch_noise, y_batch_noise = apply_noise(x_batch_pgd, model, noise_samples, max_perturbation, device, noise, test_mode)

    if y_batch_pgd.size() != torch.Size([]):
        y_batch_pgd = y_batch_pgd.repeat(noise_samples + 1)

    distances = torch.sqrt(y_batch_pgd**2 + y_batch_noise**2)
    return x_batch_noise, distances


def train_detector(model, device, train_loader, optimizer, epoch, args):
    model.train()
    for batch_idx, (x_batch, y_batch) in enumerate(train_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch, y_batch = sample_perturbed_data(x_batch, y_batch, model, 7, args.max_pgd, 7, args.max_noise, device, args.noise)
        x_batch = x_batch.detach()
        y_batch = y_batch.detach()

        # output, _ = model(x_batch)
        # pred = output.argmax(dim=1, keepdim=True)
        # plt.imshow(x_batch[-1, ...].permute(1, 2, 0).cpu(), cmap='gray')
        # plt.show()
        # print(42)

        perm = torch.randperm(len(y_batch))
        x_batch_perm, y_batch_perm = x_batch[perm, ...], y_batch[perm]
        data, target = x_batch_perm, y_batch_perm
        optimizer.zero_grad()
        _, output = model(data)
        loss = nn.MSELoss()(output[:, 0], target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Detector Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset) * 64,
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break
        if batch_idx / len(train_loader) > 0.1:
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




def test(model, device, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            if args.augment:
                data += (torch.rand(data.shape, device=device) * 2. - 1.) * 0.2
            output, _ = model(data)
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def test_renaturing(model, device, test_loader, args, apply_pgd=True):
    model.eval()
    correct = 0
    for idx, (x_batch, y_batch) in enumerate(test_loader):
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)
        x_batch, _ = sample_perturbed_data(x_batch, y_batch, model, 7 if apply_pgd else 0, args.max_pgd, 7, args.max_noise, device, args.noise, test_mode=True)
        output, _ = model(x_batch)
        pred = output.argmax(dim=1, keepdim=True).reshape(-1, len(y_batch))
        for i in range(len(y_batch)):
            frequencies = torch.bincount(pred[:, i], minlength=10)
            values, indices = torch.topk(frequencies, 2)
            if indices[0] == y_batch[i] and values[0] > values[1]:
                correct += 1
        print('agg. correct:', correct / ((idx + 1) * len(y_batch)))

def test_pgd_perturbed(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    adversarial_th_sum = 0.
    benevolent_pgd_th_sum = 0.
    pgd_th_sum = 0.
    adversarial_clean_th_sum = 0.
    benevolent_clean_th_sum = 0.
    clean_th_sum = 0.
    # with torch.no_grad():
    for batch_idx, (data, target) in enumerate(test_loader):
        data, target = data.to(device), target.to(device)
        steps = 7
        max_perturbation = 0.1
        x_perturbed = torch.zeros(data.shape).to(device)
        for i in range(len(data)):
            pgd_images = pgd_(model, device, data[None, i, :, :, :], target[None, i], steps, max_perturbation,
                              2.5 * (max_perturbation / steps), targeted=False, clip_min=0., clip_max=1., random_start=True)
            x_perturbed[i, :, :, :] = pgd_images[-1, ...]
            x_perturbed_randomized = x_perturbed + (torch.rand((10, *x_perturbed.shape), device=device) * 2. - 1.) * max_perturbation
            x_perturbed_randomized2 = torch.flatten(x_perturbed_randomized, start_dim=0, end_dim=1)
            x_perturbed_randomized2_target = target.repeat(10)

        for i in range(len(data)):
            clean_randomized = data + (torch.rand((10, *data.shape), device=device) * 2. - 1.) * max_perturbation
            clean_randomized2 = torch.flatten(clean_randomized, start_dim=0, end_dim=1)
            clean_randomized2_target = target.repeat(10)

        for i in range(len(clean_randomized2)):
            clean_randomized2[i, ...] = pgd_(model, device, clean_randomized2[None, i, ...], None, 10, max_perturbation, 2.5 * (max_perturbation / steps), clip_min=0., clip_max=1., random_start=False, regression=True)[-1, ...]
        for i in range(len(x_perturbed_randomized2)):
            x_perturbed_randomized2[i, ...] = pgd_(model, device, x_perturbed_randomized2[None, i, ...], None, 10, max_perturbation, 2.5 * (max_perturbation / steps), clip_min=0., clip_max=1., random_start=False, regression=True)[-1, ...]

        output_clean, _ = model(clean_randomized2)
        output_pgd_only, _ = model(x_perturbed)
        output, _ = model(x_perturbed_randomized2)
        test_loss += F.nll_loss(output, x_perturbed_randomized2_target, reduction='sum').item()  # sum up batch loss
        pred_pgd_only = output_pgd_only.argmax(dim=1, keepdim=True)
        pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
        individual_accuracies = torch.true_divide(torch.sum(pred.eq(x_perturbed_randomized2_target.view_as(pred)).reshape(10, -1), dim=0), 10.)
        pred_clean = output_clean.argmax(dim=1, keepdim=True)
        individual_accuracies_clean = torch.true_divide(
            torch.sum(pred_clean.eq(clean_randomized2_target.view_as(pred_clean)).reshape(10, -1), dim=0), 10.)
        was_adversarial = torch.logical_not(pred_pgd_only.eq(target.view_as(pred_pgd_only)).reshape(-1))
        # print(was_adversarial)
        # print(individual_accuracies)
        print('adversarial PGD:', individual_accuracies[was_adversarial])
        print('benevolent PGD:', individual_accuracies[torch.logical_not(was_adversarial)])
        print('clean randomized:', individual_accuracies_clean)

        print('accuracies equal?', torch.equal(individual_accuracies, individual_accuracies_clean))

        if torch.sum(was_adversarial) > 0:
            adversarial_th_sum += torch.true_divide(torch.sum(individual_accuracies[was_adversarial] > 0.5), len(individual_accuracies[was_adversarial])).cpu().item()
            adversarial_clean_th_sum += torch.true_divide(torch.sum(individual_accuracies_clean[was_adversarial] > 0.5), len(individual_accuracies_clean[was_adversarial])).cpu().item()
        benevolent_pgd_th_sum += torch.true_divide(torch.sum(individual_accuracies[torch.logical_not(was_adversarial)] > 0.5), len(individual_accuracies[torch.logical_not(was_adversarial)])).cpu().item()
        benevolent_clean_th_sum += torch.true_divide(
            torch.sum(individual_accuracies_clean[torch.logical_not(was_adversarial)] > 0.5),
            len(individual_accuracies_clean[torch.logical_not(was_adversarial)])).cpu().item()
        pgd_th_sum += torch.true_divide(torch.sum(individual_accuracies > 0.5),
                                          len(individual_accuracies)).cpu().item()
        clean_th_sum += torch.true_divide(torch.sum(individual_accuracies_clean > 0.5), len(individual_accuracies_clean)).cpu().item()
        print('agg. adversarial PGD over th.:', adversarial_th_sum / (batch_idx + 1))
        print('agg. benevolent PDG over th.:', benevolent_pgd_th_sum / (batch_idx + 1))
        print('agg. PDG over th.:', pgd_th_sum / (batch_idx + 1))
        print('agg. adversarial clean over th.:', adversarial_clean_th_sum / (batch_idx + 1))
        print('agg. benevolent clean over th.:', benevolent_clean_th_sum / (batch_idx + 1))
        print('agg. clean randomized over th.:', clean_th_sum / (batch_idx + 1))

        new_correct = pred.eq(x_perturbed_randomized2_target.view_as(pred)).sum().item()
        correct += new_correct
        # print(new_correct / len(x_perturbed_randomized2_target))

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


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
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--load-model', action='store_true', default=True,
                        help='For Loading the last Model')
    parser.add_argument('--augment', action='store_true', default=False,
                        help='Whether data should be perturbed when training the classifier (default: False)')
    parser.add_argument('--retrain-detector', action='store_true', default=False,
                        help='Retrain the detector model')
    parser.add_argument('--noise', default='uniform')
    parser.add_argument('--max-pgd', type=float, default=0.1)
    parser.add_argument('--max-noise', type=float, default=0.1)

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
        model.load_state_dict(torch.load("mnist_cnn_std.pt", map_location=device))
    else:
        scheduler = StepLR(optimizer_classifier, step_size=1, gamma=args.gamma)
        for epoch in range(1, args.epochs + 1):
            train_classifier(args, model, device, train_loader, optimizer_classifier, epoch)
            test(model, device, test_loader, args)
            scheduler.step()
        if args.save_model:
            torch.save(model.state_dict(), "mnist_cnn_std.pt")


    # save_ad_examples(model, device, train_loader)


    scheduler = ReduceLROnPlateau(optimizer_detector, patience=2)
    # for epoch in range(1, args.epochs + 1):
    #     train_detector(model, device, train_loader, optimizer_detector, epoch, args)
    if args.retrain_detector:
        train_detector(model, device, train_loader, optimizer_detector, 0, args)
        torch.save(model.state_dict(), "mnist_cnn_detector.pt")
    else:
        model.load_state_dict(torch.load("mnist_cnn_detector.pt"))

    # test_pgd_perturbed(model, device, test_loader)
    test_renaturing(model, device, test_loader, args, apply_pgd=True)

if __name__ == '__main__':
    main()