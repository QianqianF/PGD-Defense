# evaluate base classifier entropy

import argparse
import datetime
import os
import numpy as np
from time import time

from architectures import get_architecture
from intermediate import Intermediate
from core import Smooth
from datasets import get_dataset, DATASETS, get_num_classes
import torch
from torch.profiler import profile, record_function, ProfilerActivity

parser = argparse.ArgumentParser(description='Compare entropies of clean vs perturbed images')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")

parser.add_argument("--sample_size", type=int, default=10)
args = parser.parse_args()

def entropy(logits):
	return torch.sum(torch.sum(torch.log2(F.softmax(logits, 1)), dim=1))


if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])

    # create the test dataset
    # train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')

    pin_memory = False

    # labelled_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
    #                   num_workers=args.workers, pin_memory=pin_memory)
    # train_loader = MultiDatasetsDataLoader([labelled_loader])
    # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
    #                          num_workers=args.workers, pin_memory=pin_memory)

    results = []
    for i in range(len(test_dataset)):

        (x, label) = dataset[i]
        x = x.cuda()

        logits = base_classifier(x)
        clean_entropy = entropy(logits)

        noise_entropy = torch.zeros(args.sample_size)
        for i in args.sample_size:
            noise = torch.randn_like(x, device='cuda') * self.sigma
            logits = self.base_classifier(x + noise)
            noise_entropy[i] = entropy(logits)

        diff = (noise_entropy - clean_entropy).mean()

        res = {"clean_entropy": clean_entropy, "noise_entropy": noise_entropy, "diff": diff}
        results.append(res)
        print(res)

    np.save('entropy', results)



        

    

