# evaluate base classifier entropy

import argparse
import datetime
import os
import numpy as np
from time import time

import torch.nn.functional as F
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
    class_probabilities = F.softmax(logits, 1)
	return -torch.sum(class_probabilities * torch.log2(class_probabilities), dim=1)


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
        
        with torch.no_grad():
            (x, label) = test_dataset[i]
            x = x[None, :].cuda()

            _, channel, height, width = x.shape
            batch = torch.zeros((args.sample_size+1, channel, height, width)).cuda()
            batch[0] = x

            for i in range(args.sample_size):
                noise = torch.randn_like(x, device='cuda') * args.sigma
                batch[i+1] = x + noise

            logits = base_classifier(batch)
            #print(logits.shape)
            #clean_entropy = entropy(logits[0])

            #noise_entropy = torch.zeros(args.sample_size)
            #for i in range(args.sample_size):
            #    noise_entropy[i] = entropy(logits[i+1])
            
            batch_entropy = entropy(logits)
            clean_entropy = batch_entropy[0]
            noise_entropy = batch_entropy[1:]
            diff = (noise_entropy - clean_entropy).mean()

            res = {"clean_entropy": clean_entropy, "noise_entropy": noise_entropy, "diff": diff}
            results.append(res)
            #print(res)

    np.save('entropy', results)



        

    

