'''
Script to visualize YOLOv2's performances on custom data.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import argparse
import os
import numpy as np
import imageio

from config import Config
from utils import *

#########################################################
################### Parsing arguments ###################
#########################################################

argparser = argparse.ArgumentParser(
    description="Visualize the detection system's performances on custom data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to the folder containing the data and ground truth annotation.",
    default=os.path.join('pelvis_scan','data','train'))

argparser.add_argument(
    '-a',
    '--annot_file',
    help="ground truth annotation file name.",
    default='annotations_train.p')

argparser.add_argument(
    '-p',
    '--prediction_file',
    help="path to the prediction file.",
    default=os.path.join('results','predictions','pred_boxes.p'))

argparser.add_argument(
    '-r',
    '--results_dir',
    help="path to the folder where the data files come from and where the results are going to be stored.",
    default='results')

argparser.add_argument(
    '-V',
    '--visualize',
    help="enables to save all images with ground truth and predicted bounding boxes.",
    action='store_true')

argparser.add_argument(
    '-H',
    '--history',
    help="enables to plot history graphs of the loss functions.",
    action='store_true')

argparser.add_argument(
    '-hp',
    '--history_path',
    help="path to the folder where the history files are stored.",
    default=os.path.join('results','history'))

########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    data_path = args.data_path
    gt_annot = args.annot_file
    pred_annot = args.prediction_file
    results_dir = args.results_dir
    visualize = args.visualize
    h_mode = args.history
    history_path = args.history_path

    # Creating config instance
    config = Config()
    ratio = (config.INPUT_DIM[0]/config.OUTPUT_DIM[0])

    # Drawing bounding boxes
    if visualize:
        gt = load_annotation(os.path.join(data_path,gt_annot))
        pred = load_annotation(pred_annot)
        dir_list = [x for x in sorted(os.listdir(data_path)) if x.endswith('-image.jpg')]
        for filename in dir_list:
            print('Drawning bounding box for {}'.format(filename))
            to_draw = pred[filename].keys()
            draw_bb(os.path.join(data_path,filename),gt[filename]['bb'],to_draw=to_draw,save=True,results_path=os.path.join(results_dir,'images'))
            draw_bb(os.path.join(data_path,filename),pred[filename],to_draw=to_draw,save=True,results_path=os.path.join(results_dir,'images'),rescale=True,ratio=ratio,colors=['lime','deeppink','cyan','gold'])
            over_draw(os.path.join(data_path,filename),gt[filename]['bb'],pred[filename],to_draw=to_draw,save=True,results_path=os.path.join(results_dir,'images'),ratio=ratio)


    # Plotting history
    if h_mode:
        # Computed arguments
        graphs_dir = os.path.join(results_dir,'graphs')
        if not os.path.exists(graphs_dir):
            os.makedirs(graphs_dir)

        history = merge_history(history_path)
        metrics = ['loss','classification_loss','coord_loss','conf_loss']
        legend = ['loss (total)', 'classification loss','coordinates loss','confidence loss']
        save = True
        for i in range(len(metrics)):
            learning_graph(history,[metrics[i],'val_'+metrics[i]],['training '+legend[i],'validation '+legend[i] ],save=save,path=os.path.join(graphs_dir,metrics[i]+'_lin.pdf'),scale='linear')
            learning_graph(history,[metrics[i],'val_'+metrics[i]],['training '+legend[i],'validation '+legend[i] ],save=save,path=os.path.join(graphs_dir,metrics[i]+'_log.pdf'),scale='log')



############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
