'''
Script to run k-mean clustering on custom data to generate custom anchor dimensions.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import argparse
import os
import numpy as np
import shutil
import pickle
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

from config import Config
from utils import load_annotation,save_annotation


#########################################################
################### Parsing arguments ###################
#########################################################

argparser = argparse.ArgumentParser(
    description="Transforms an image dataset into a npz file usable by yolo.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to the file containing all the bounding boxes annotations.",
    default=os.path.join('pelvis_scan','2D_DATASET_CT_FULL','train','annotations_train.p'))

########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    data_path = args.data_path

    # Creating config instance
    config = Config()

    # K-mean clustering
    input_size = config.INPUT_DIM[0]
    k = 5
    S = 13
    data_path = '/Volumes/LUCA_EHD/TFE_DATA/USABLE_DATASETS/2D_DATASET_CT/2D_annotations_CT.p'
    print(k_mean_clustering(k,data_path,S,input_size,display=True))

def k_mean_clustering(k,data_path,S,input_size,display=False):
    dict = load_annotation(data_path)
    h = []
    w = []
    for file in dict.keys():
        for organ in dict[file]['bb'].keys():
            box = dict[file]['bb'][organ]
            if box is not None and file.startswith('charleroi'):
                assert len(box) == 4
                h.append(box[2]-box[0])
                w.append(box[3]-box[1])

    # data
    df = pd.DataFrame({'x' : w, 'y' : h})

    # fit
    kmeans = KMeans(n_clusters=k)
    kmeans.fit(df)

    # learn the labels
    labels = kmeans.predict(df)
    centroids = kmeans.cluster_centers_

    # display
    if(display):
        colmap = {1: 'r', 2: 'g', 3: 'b',4: 'y',5: 'm'}
        fig = plt.figure(figsize=(5, 5))
        colors = list(map(lambda x: colmap[x+1], labels))

        plt.scatter(df['x'], df['y'], color=colors, alpha=0.5, edgecolor='k')
        for idx, centroid in enumerate(centroids):
            plt.scatter(*centroid, color=colmap[idx+1],label='cluster '+str(idx))
        plt.xlim(0, 130)
        plt.ylim(0, 120)
        plt.legend()
        plt.ylabel('height (pixels)')
        plt.xlabel('width (pixels)')
        plt.show()

    return (centroids/input_size)*S

########################################################
######################### Main #########################
########################################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
