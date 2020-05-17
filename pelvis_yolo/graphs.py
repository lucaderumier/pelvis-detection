'''
Script to plot graphs of YOLOv2's performances on custom data.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import argparse
import os
import numpy as np
import re

from config import Config
from utils import *

#########################################################
################### Parsing arguments ###################
#########################################################

argparser = argparse.ArgumentParser(
    description="Plot graphs of the detection system's performances on custom data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to the folder containing the stages folders.",
    default='stages')

########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    data_path = args.data_path

    # Config
    config = Config()

    # Iterates through the folder to generate iou dictionnary
    sets = [x for x in ['train','val','test'] if x in os.listdir(data_path)]
    iou = {}
    stats = {}
    for set in sets:
        folders = [x for x in os.listdir(os.path.join(data_path,set)) if x.startswith('stage')]
        stages = [int(re.findall('\d+', x)[0]) for x in folders]
        iou.update({set : {}})
        stats.update({set : {}})
        for i in range(len(folders)):
            iou_stage = load_annotation(os.path.join(data_path,set,folders[i],'stats','iou.p'))
            stats_stage = load_annotation(os.path.join(data_path,set,folders[i],'stats','classification.p'))
            iou[set].update({stages[i] : {'mean': iou_stage['mean'] , 'variance': iou_stage['variance']}})
            stats[set].update({stages[i] : stats_stage})

    if not os.path.exists(os.path.join(data_path,'graphs')):
        os.makedirs(os.path.join(data_path,'graphs'))

    save_annotation(iou,os.path.join(data_path,'graphs','iou_stages.p'))
    save_annotation(stats,os.path.join(data_path,'graphs','stats_stages.p'))

    # Plotting part
    epochs = [(x+1)*config.EPOCHS for x in sorted(stages)]
    #plot_iou(iou,epochs,data_path,save=False)
    plot_iou(iou,epochs,data_path,save=True)
    #plot_stats(stats,epochs,data_path,save=False)
    plot_stats(stats,epochs,data_path,save=True)

####################################################
################## Plot functions ##################
####################################################

def plot_stats(stats,epochs,data_path,save=False):
    '''Plots the precision, recall and false negative rate graphs for the sets (training, validation at least).

    Inputs:
        stats: a dictionnary that contains the stats.
            stats = {'train' : { 0 : {'bladder' : {'P' : {'mean' : .. 'variance' : ...} , 'R' : {...}},
                                     'rectum' : ...},
                               1 : {....}, ..}
                    'val' : .... }
    '''
    precision = np.zeros((len(stats),len(epochs)))
    precision_var = np.zeros((len(stats),len(epochs)))
    recall = np.zeros((len(stats),len(epochs)))
    recall_var = np.zeros((len(stats),len(epochs)))
    fnrate = np.zeros((len(stats),len(epochs)))
    fnrate_var = np.zeros((len(stats),len(epochs)))

    for set,stages in stats.items():
        if set == 'train':
            i = 0
        elif set == 'val':
            i = 1
        else:
            raise ValueError('unexpected set value.')

        for stage,organs in stages.items():
            precision[i][stage] = nan_to_x((organs['bladder']['P']['mean']+organs['rectum']['P']['mean']+organs['prostate']['P']['mean'])/3)
            precision_var[i][stage] = nan_to_x((organs['bladder']['P']['variance']+organs['rectum']['P']['variance']+organs['prostate']['P']['variance'])/3)
            recall[i][stage] = nan_to_x((organs['bladder']['R']['mean']+organs['rectum']['R']['mean']+organs['prostate']['R']['mean'])/3)
            recall_var[i][stage] = nan_to_x((organs['bladder']['R']['variance']+organs['rectum']['R']['variance']+organs['prostate']['R']['variance'])/3)
            fnrate[i][stage] = nan_to_x((organs['bladder']['FNR']['mean']+organs['rectum']['FNR']['mean']+organs['prostate']['FNR']['mean'])/3,x=1)
            fnrate_var[i][stage] = nan_to_x((organs['bladder']['FNR']['variance']+organs['rectum']['FNR']['variance']+organs['prostate']['FNR']['variance'])/3)

    path = os.path.join(data_path,'graphs')

    if save:
        generic_graph([precision[0],precision[1]],[precision_var[0],precision_var[1]],epochs,['training precision','validation precision'],ylabel='precision',title='Training vs Validation Precision',save=True,path=os.path.join(path,'prec_tvsv.png'))
        generic_graph([recall[0],recall[1]],[recall_var[0],recall_var[1]],epochs,['training recall','validation recall'],title='Training vs Validation Recall',ylabel='recall',save=True,path=os.path.join(path,'rec_tvsv.png'))
        generic_graph([fnrate[0],fnrate[1]],[fnrate_var[0],fnrate_var[1]],epochs,['training false negative rate','validation false negative rate'],ylabel='false negative rate',title='Training vs Validation False Negative Rate',save=True,path=os.path.join(path,'fnr_tvsv.png'))
    else:
        generic_graph([precision[0],precision[1]],[precision_var[0],precision_var[1]],epochs,['training precision','validation precision'],ylabel='precision',title='Training vs Validation Precision')
        generic_graph([recall[0],recall[1]],[recall_var[0],recall_var[1]],epochs,['training recall','validation recall'],ylabel='recall',title='Training vs Validation Recall')
        generic_graph([fnrate[0],fnrate[1]],[fnrate_var[0],fnrate_var[1]],epochs,['training false negative rate','validation false negative rate'],ylabel='false negative rate',title='Training vs Validation False Negative Rate')





def plot_iou(iou,epochs,data_path,save=False):
    '''Plots the mean iou graphs for the sets (training, validation at least)
       along with iou for all organs separately.

    Inputs:
        iou: a dictionnary that contains the iou.
            iou = {'train' : { 0 : {'mean' : {'bladder' : .., 'rectum' : .., 'prostate' : ..},
                                    'variance': {.....}}},
                               1 : {....}, ..
                    'val' : .... }
    '''
    # Defining arrays for mean and variance of each organs and sets.
    bladder_mean = np.zeros((len(iou),len(epochs)))
    bladder_var = np.zeros((len(iou),len(epochs)))
    rectum_mean = np.zeros((len(iou),len(epochs)))
    rectum_var = np.zeros((len(iou),len(epochs)))
    prostate_mean = np.zeros((len(iou),len(epochs)))
    prostate_var = np.zeros((len(iou),len(epochs)))
    total_mean = np.zeros((len(iou),len(epochs)))
    total_var = np.zeros((len(iou),len(epochs)))

    for set,stages in iou.items():
        if set == 'train':
            i = 0
        elif set == 'val':
            i = 1
        else:
            raise ValueError('unexpected set value.')
        for stage in stages.keys():
            bladder_mean[i][stage] = nan_to_x(stages[stage]['mean']['bladder'])
            bladder_var[i][stage] = nan_to_x(stages[stage]['variance']['bladder'])
            rectum_mean[i][stage] = nan_to_x(stages[stage]['mean']['rectum'])
            rectum_var[i][stage] = nan_to_x(stages[stage]['variance']['rectum'])
            prostate_mean[i][stage] = nan_to_x(stages[stage]['mean']['prostate'])
            prostate_var[i][stage] = nan_to_x(stages[stage]['variance']['prostate'])
            total_mean[i][stage] = (bladder_mean[i][stage]+rectum_mean[i][stage]+prostate_mean[i][stage])/3
            total_var[i][stage] = (bladder_var[i][stage]+rectum_var[i][stage]+prostate_var[i][stage])/3

    path = os.path.join(data_path,'graphs')

    if save:
        generic_graph([total_mean[0],total_mean[1]],[total_var[0],total_var[1]],epochs,['training mean','validation mean'],title='Training vs Validation IoU',save=True,path=os.path.join(path,'iou_tvsv.png'))
        generic_graph([bladder_mean[0],rectum_mean[0],prostate_mean[0]],[bladder_var[0],rectum_var[0],prostate_var[0]],epochs,['bladder','rectum','prostate'],title='Training IoU',save=True,path=os.path.join(path,'iou_training_all.png'))
        generic_graph([bladder_mean[1],rectum_mean[1],prostate_mean[1]],[bladder_var[1],rectum_var[1],prostate_var[1]],epochs,['bladder','rectum','prostate'],title='Validation  IoU',save=True,path=os.path.join(path,'iou_validation_all.png'))
    else:
        generic_graph([total_mean[0],total_mean[1]],[total_var[0],total_var[1]],epochs,['training mean','validation mean'],title='Training vs Validation IoU')
        generic_graph([bladder_mean[0],rectum_mean[0],prostate_mean[0]],[bladder_var[0],rectum_var[0],prostate_var[0]],epochs,['bladder','rectum','prostate'],title='Training IoU')
        generic_graph([bladder_mean[1],rectum_mean[1],prostate_mean[1]],[bladder_var[1],rectum_var[1],prostate_var[1]],epochs,['bladder','rectum','prostate'],title='Validation  IoU')

############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
