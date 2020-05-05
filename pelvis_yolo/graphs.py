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

########################################################
#################### GPU Constraint ####################
########################################################

gpu = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

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
    for set in sets:
        folders = [x for x in os.listdir(os.path.join(data_path,set)) if x.startswith('stage')]
        stages = [int(re.findall('\d+', x)[0]) for x in folders]
        iou.update({set : {}})
        for i in range(len(folders)):
            iou_stage = load_annotation(os.path.join(data_path,set,folders[i],'stats','iou.p'))
            iou[set].update({stages[i] : {'mean': iou_stage['mean'] , 'variance': iou_stage['variance']}})

    if not os.path.exists(os.path.join(data_path,'graphs')):
        os.makedirs(os.path.join(data_path,'graphs'))

    save_annotation(iou,os.path.join(data_path,'graphs','iou_stages.p'))

    # Plotting part
    epochs = [(x+1)*config.EPOCHS for x in sorted(stages)]
    plot_iou(iou,epochs,data_path,save=False)

####################################################
################## Plot functions ##################
####################################################

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
            bladder_mean[i][stage] = nan_to_zero(stages[stage]['mean']['bladder'])
            bladder_var[i][stage] = nan_to_zero(stages[stage]['variance']['bladder'])
            rectum_mean[i][stage] = nan_to_zero(stages[stage]['mean']['rectum'])
            rectum_var[i][stage] = nan_to_zero(stages[stage]['variance']['rectum'])
            prostate_mean[i][stage] = nan_to_zero(stages[stage]['mean']['prostate'])
            prostate_var[i][stage] = nan_to_zero(stages[stage]['variance']['prostate'])
            total_mean[i][stage] = (bladder_mean[i][stage]+rectum_mean[i][stage]+prostate_mean[i][stage])/3
            total_var[i][stage] = (bladder_var[i][stage]+rectum_var[i][stage]+prostate_var[i][stage])/3

    path = os.path.join(data_path,'graphs')

    if save:
        iou_graph([total_mean[0],total_mean[1]],[total_var[0],total_var[1]],epochs,['training mean','validation mean'],title='Training vs Validation IoU',save=True,path=os.path.join(path,'iou_tvsv.png'))
        iou_graph([bladder_mean[0],rectum_mean[0],prostate_mean[0]],[bladder_var[0],rectum_var[0],prostate_var[0]],epochs,['bladder','rectum','prostate'],title='Training IoU',save=True,path=os.path.join(path,'iou_training_all.png'))
        iou_graph([bladder_mean[1],rectum_mean[1],prostate_mean[1]],[bladder_var[1],rectum_var[1],prostate_var[1]],epochs,['bladder','rectum','prostate'],title='Validation  IoU',save=True,path=os.path.join(path,'iou_validation_all.png'))
    else:
        iou_graph([total_mean[0],total_mean[1]],[total_var[0],total_var[1]],epochs,['training mean','validation mean'],title='Training vs Validation IoU')
        iou_graph([bladder_mean[0],rectum_mean[0],prostate_mean[0]],[bladder_var[0],rectum_var[0],prostate_var[0]],epochs,['bladder','rectum','prostate'],title='Training IoU')
        iou_graph([bladder_mean[1],rectum_mean[1],prostate_mean[1]],[bladder_var[1],rectum_var[1],prostate_var[1]],epochs,['bladder','rectum','prostate'],title='Validation  IoU')
############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
