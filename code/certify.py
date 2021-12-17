# evaluate a smoothed classifier on a dataset
import argparse
import datetime
import os
from time import time

from architectures import get_architecture
from intermediate import Intermediate
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
import torch
from torch.profiler import profile, record_function, ProfilerActivity


parser = argparse.ArgumentParser(description='Certify many examples')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outfile", type=str, help="output file")
parser.add_argument("--batch", type=int, default=1000, help="batch size")
parser.add_argument("--skip", type=int, default=1, help="how many examples to skip")
parser.add_argument("--max", type=int, default=-1, help="stop after this many examples")
parser.add_argument("--split", choices=["train", "test"], default="test", help="train or test set")
parser.add_argument("--N0", type=int, default=100)
parser.add_argument("--N", type=int, default=100000, help="number of samples to use")
parser.add_argument("--alpha", type=float, default=0.001, help="failure probability")

parser.add_argument('--use-intermediate', action='store_true', help="use intermediate PGD layer for certification")
parser.add_argument("--pgd-steps", type=int, default=5, help="number of PGD steps in intermediate layer")
parser.add_argument("--pgd-epsilon", type=float, default=0.1, help="maximum l2 perturbation applied to samples in the intermediate layer")
parser.add_argument("--pgd-bnn-samples", type=int, default=1, help="BNN samples to take at each step of the entropy attack (leave at 1 for plain NNs)")

parser.add_argument("--profile", action='store_true')
args = parser.parse_args()

if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the smooothed classifier
    if args.use_intermediate:
        smoothed_classifier = Smooth(Intermediate(base_classifier, args.pgd_steps, args.pgd_epsilon, args.pgd_bnn_samples), get_num_classes(args.dataset), args.sigma)
    else:
        smoothed_classifier = Smooth(base_classifier, get_num_classes(args.dataset), args.sigma)

    # prepare output file
    f = open(args.outfile, 'w')
    if not args.profile:
        print("idx\tlabel\tpredict\tradius\tcorrect\ttime", file=f, flush=True)

    # iterate through the dataset
    dataset = get_dataset(args.dataset, args.split)
    for i in range(len(dataset)):

        # only certify every args.skip examples, and stop after args.max examples
        if i % args.skip != 0:
            continue
        if i == args.max:
            break

        (x, label) = dataset[i]
        x = x.cuda()

        if args.profile:
            print("starting profiling...")
            with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA], record_shapes=True) as prof:
                with record_function("model_certification"):
                    smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
            
            print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=50))
            break
                
        before_time = time()
        # certify the prediction of g around x
        prediction, radius = smoothed_classifier.certify(x, args.N0, args.N, args.alpha, args.batch)
        after_time = time()
        correct = int(prediction == label)

        time_elapsed = str(datetime.timedelta(seconds=(after_time - before_time)))
        print("{}\t{}\t{}\t{:.3}\t{}\t{}".format(
            i, label, prediction, radius, correct, time_elapsed), file=f, flush=True)

    f.close()
