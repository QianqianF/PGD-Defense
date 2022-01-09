# evaluate a smoothed classifier on a dataset
import argparse
import datetime
import os
from time import time

from architectures import get_architecture
from ece import ece
from core import Smooth
from swag import SWAGDiagonalModel
from datasets import get_dataset, DATASETS, get_num_classes
import torch
from torch.utils.data.dataloader import DataLoader
import numpy as np


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument('--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument("--samples", default=8, type=int, help="samples of the BNN")

parser.add_argument("--num-aug", default=5, type=int, help="number of data aug samples per each clean image")
parser.add_argument("--sigma", default=0.12, type=int, help="sigma of data aug")
parser.add_argument("--swag", action="store_true")
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    if args.swag:
        base_classifier = SWAGDiagonalModel(base_classifier, 'cuda')
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # prepare output file
    f = open(args.outfile, 'w')
    print("batch_idx\taccuracy\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    pin_memory = (args.dataset == "imagenet")
    loader = DataLoader(dataset, shuffle=True, batch_size=args.batch,
                      num_workers=args.workers, pin_memory=pin_memory)
    probability_batches = []
    label_batches = []
    overall_accuracy = 0.
    for i, batch in enumerate(loader):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = batch

        before_time = time()
        # certify the prediction of g around x
        with torch.no_grad():
            x = x.cuda()
            label = label.repeat(args.num_aug+1).cuda()
            batch = x.clone()

            for j in range(args.num_aug):
                noise = torch.randn_like(x, device='cuda') * args.sigma
                batch = torch.cat((batch, x + noise))

            pred = 0.
            if args.swag:
                pred = base_classifier(args.samples, batch)
            else:
                for j in range(args.samples):
                    pred += torch.nn.functional.softmax(base_classifier(batch), dim=1)
                pred /= args.samples

        probability_batches.append(pred.cpu().numpy())
        label_batches.append(label.cpu().numpy())
        after_time = time()
        accuracy = torch.sum(torch.argmax(pred, axis=1) == label) / label.shape[0]
        overall_accuracy += float(accuracy.cpu().item()) * label.shape[0]

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{:.3}\t{}".format(
            i, accuracy, time_elapsed), file=f, flush=True)

    predicted_probabilities = np.concatenate(probability_batches, axis=0)
    assert isinstance(predicted_probabilities, np.ndarray)
    assert predicted_probabilities.ndim == 2 and predicted_probabilities.shape[1] == 10
    assert np.allclose(np.sum(predicted_probabilities, axis=1), 1.0)
    actual_classes = np.concatenate(label_batches, axis=0)

    ece_score = ece(predicted_probabilities, actual_classes)
    overall_accuracy /= predicted_probabilities.shape[0]
    print("ECE score: {:.3f}".format(ece_score), file=f, flush=True)
    print("Overall Accuracy: {:3f}".format(overall_accuracy), file=f, flush=True)

    f.close()
