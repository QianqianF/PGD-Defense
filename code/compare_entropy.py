# evaluate base classifier entropy and plot entropy

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
from attacks import PGD_L2

import pandas as pd
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='Compare entropies of clean vs perturbed images')
parser.add_argument("dataset", choices=DATASETS, help="which dataset")
parser.add_argument("base_classifier", type=str, help="path to saved pytorch model of base classifier")
parser.add_argument("sigma", type=float, help="noise hyperparameter")
parser.add_argument("outdir", type=str, default="entropy_results")

parser.add_argument("--sample_size", type=int, default=5)
parser.add_argument('--bbb-samples', type=int, default=1) 
args = parser.parse_args()

def entropy(class_probabilities):
    return -torch.sum(torch.nan_to_num(class_probabilities * torch.log2(class_probabilities)), dim=1)

def plot_entropy(results, plotname):
    entropy_table = pd.DataFrame.from_records(results)
    entropy_table['clean_entropy'] = entropy_table['clean_entropy'].transform(lambda t: t.item())
    entropy_table['mean_noise_entropy'] = entropy_table['noise_entropy']\
                                      .transform(lambda l: np.array([t.cpu() for t in l]).mean())
    entropy_table['diff'] = entropy_table['diff'].transform(lambda t: t.item())
    plot = entropy_table[['clean_entropy', 'mean_noise_entropy']].plot(kind='hist', bins=20, \
                                                            alpha=0.7, color=['#A0E8AF', '#FFCF56'])
    fig = plot.get_figure()
    fig.savefig(os.path.join(args.outdir, plotname))

def showImage(im, imName):
    plt.imshow(im)
    plt.savefig(os.path.join(args.outdir, imName))


if __name__ == "__main__":
    # load the base classifier
    checkpoint = torch.load(args.base_classifier)
    base_classifier = get_architecture(checkpoint["arch"], args.dataset)
    base_classifier.load_state_dict(checkpoint['state_dict'])
    classifier = Intermediate(base_classifier=base_classifier, num_steps=1)

    # create the test dataset
    # train_dataset = get_dataset(args.dataset, 'train')
    test_dataset = get_dataset(args.dataset, 'test')

    pin_memory = False

    # labelled_loader = DataLoader(train_dataset, shuffle=True, batch_size=args.batch,
    #                   num_workers=args.workers, pin_memory=pin_memory)
    # train_loader = MultiDatasetsDataLoader([labelled_loader])
    # test_loader = DataLoader(test_dataset, shuffle=False, batch_size=args.batch,
    #                          num_workers=args.workers, pin_memory=pin_memory)
    
    os.makedirs(args.outdir, exist_ok = True)
    results = []
    results_after = []
    f = open(os.path.join(args.outdir, "entropy.txt"), 'w+')
    print("idx\tclean_entropy\tnoise_entropy\tdiff", file=f, flush=True)
    for i in range(len(test_dataset)):
        
        with torch.no_grad():
            (x, label) = test_dataset[i]
            x = x[None, :].cuda()
            label = torch.tensor([label] * (args.sample_size + 1)).cuda()

            _, channel, height, width = x.shape
            batch = torch.zeros((args.sample_size+1, channel, height, width)).cuda()
            batch[0] = x

            for j in range(args.sample_size):
                noise = torch.randn_like(x, device='cuda') * args.sigma
                batch[j+1] = x + noise

            if args.bbb_samples > 1:
                _, class_probabilities = base_classifier.sample_elbo_with_output(batch,label, torch.nn.CrossEntropyLoss(), args.bbb_samples)
            else:
                logits = base_classifier(batch)
                class_probabilities = F.softmax(logits, 1)

            batch_entropy = entropy(class_probabilities)
            clean_entropy = batch_entropy[0]
            noise_entropy = batch_entropy[1:]
            diff = (noise_entropy - clean_entropy).mean()

            _, class_probabilities_after = classifier(batch)
            batch_entropy_after = entropy(class_probabilities_after)
            clean_entropy_after = batch_entropy[0]
            noise_entropy_after = batch_entropy[1:]
            diff_after = (clean_entropy_after - noise_entropy_after).mean()
            

            print("{}\t{}\t{}\t{}".format(i, clean_entropy, noise_entropy, diff), file=f, flush=True)
            res = {"clean_entropy": clean_entropy, "noise_entropy": noise_entropy, "diff": diff}
            res_after = {"clean_entropy": clean_entropy, "noise_entropy": noise_entropy, "diff": diff}
            results.append(res)
            results_after.append(res_after)

             # visualize image before and after for the first 5 images
            if i < 5:
                batch = torch.clip(batch, 0., 1.)
                showImage(batch[0], "image_" + i + "_clean.png")
                for j in range(args.sample_size):
                    showImage(batch[j+1], "image_" + i + "_noise_j.png")

                attacker = PGD_L2(steps=classifier.num_steps, device='cuda', max_norm=classifier.epsilon)
                high_confidence_image = attacker.attack(classifier.base_classifier, x, None, entropy_attack=True, entropy_samples=classifier.entropy_samples)
                
                showImage(high_confidence_image[0], "image_" + i + "_clean_after.png")
                for j in range(args.sample_size):
                    showImage(high_confidence_image[j+1], "image_" + i + "_noise_" + j + "_after.png")


            

    np.save(os.path.join(args.outdir, "entropy.npy"), results)
    np.save(os.path.join(args.outdir, "entropy_after.npy"), results_after)
    
    # Plotting
    plot_entropy(results, "clean_vs_mean_entropy.png")
    plot_entropy(results_after, "clean_vs_mean_entropy_after.png")
    
        

    

