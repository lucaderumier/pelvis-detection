'''
Script to evaluate YOLOv2's performances on custom data.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import argparse
import os
import numpy as np
#import imageio
import pickle

from config import Config
from utils import *

#########################################################
################### Parsing arguments ###################
#########################################################

argparser = argparse.ArgumentParser(
    description="Evaluates the detection system on custom data.")

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
    '-r',
    '--results_dir',
    help="path to the folder where the data files come from and where the results are going to be stored.",
    default='results')


########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    data_path = args.data_path
    gt_annot = args.annot_file
    results_dir = args.results_dir


    # Creating config instance
    config = Config()
    ratio = (config.INPUT_DIM[0]/config.OUTPUT_DIM[0])

    # Creating missing directories
    if not os.path.exists(os.path.join(results_dir,'stats')):
        os.makedirs(os.path.join(results_dir,'stats'))

    # Loading files
    print(os.path.join(data_path,gt_annot))
    gt = load_annotation(os.path.join(data_path,gt_annot))
    pred = load_annotation(os.path.join(results_dir,'predictions','pred_boxes.p'))

    # Computing statistics
    found_stats = found(gt,pred)
    save_annotation(found_stats,os.path.join(results_dir,'stats','found.p'))

    iou_stats = IoU(gt,pred,ratio)
    save_annotation(iou_stats,os.path.join(results_dir,'stats','iou.p'))

    classification_stats = classification(gt,pred,config.INPUT_DIM,ratio)
    save_annotation(classification_stats,os.path.join(results_dir,'stats','classification.p'))

    cases_stats = cases(iou_stats)
    save_annotation(cases_stats,os.path.join(results_dir,'stats','cases.p'))

    # Writing directories to a file
    f = open(os.path.join(results_dir,'stats','stats.txt'),'w+')
    f.write('{}\n\n'.format(found_stats))
    f.write('mean : {}\n variance : {}\n\n'.format(iou_stats['mean'],iou_stats['variance']))
    f.write('{}\n\n'.format(classification_stats))
    f.write('{}\n\n'.format(cases_stats))
    f.close()



##########################################################
################## Statistics functions ##################
##########################################################

def found(gt,pred):
    '''Identifies the percentage of organs that were accurately
    or inaccurately found and missed by the detection system.

    Inputs:
        gt: ground truth annotations.
        pred: predicted annotations.

    Returns:
        percentages: a dictionnary that contains for each organ the statistics.
                     percentages = {'bladder': {'found' : ..,
                                               'missed' : ..,
                                               'found_noorg' : (only if total_noorg > 0),
                                               'missed_noorg' : (only if total_noorg > 0),
                                               'additional' : ..,
                                               'total': ...,
                                               'total_noorg': 0},...}
    '''

    # Util variable
    start = True

    # Iterating through the dictionnary files
    for file,organs in pred.items():

        # Initialize the statistics
        if start:
            stats = {}
            for o in organs.keys():
                stats.update({o : {'found' : 0, 'missed' : 0,'found_noorg' : 0,'missed_noorg' : 0, 'additional' : 0, 'total': 0, 'total_noorg': 0}})
            start = False

        # Verify that both gt and pred contain this file
        assert file in gt.keys()

        # Iterate through the organs of file
        for organ in organs.keys():

            # The organ is not in the gt image
            if organ not in gt[file]['bb'].keys():
                stats[organ]['total_noorg'] += 1
                # The system did not detect one (good)
                if(len(pred[file][organ]) == 0):
                    stats[organ]['found_noorg'] += 1
                # The system detected one anyway (bad)
                if(len(pred[file][organ]) > 0):
                    stats[organ]['missed_noorg'] += 1
                # The system detected more than one (very bad)
                if(len(pred[file][organ]) > 1):
                    stats[organ]['additional'] += 1

            # The organ is in the gt image
            else:
                stats[organ]['total'] += 1
                # The system did not detect one (very bad)
                if(len(pred[file][organ]) == 0):
                    stats[organ]['missed'] += 1
                # The system detected one (good)
                if(len(pred[file][organ]) > 0):
                    stats[organ]['found'] += 1
                # The system detected more than one (not so good)
                if(len(pred[file][organ]) > 1):
                    stats[organ]['additional'] += 1

    # Computing the averages
    percentages = {}
    for organ in stats.keys():

        # Some of the images dont have this organ
        if stats[organ]['total_noorg'] > 0:
            percentages.update({organ : {
                        'found' : (stats[organ]['found']/stats[organ]['total'])*100,
                        'missed' : (stats[organ]['missed']/stats[organ]['total'])*100,
                        'found_noorg' : (stats[organ]['found_noorg']/stats[organ]['total_noorg'])*100,
                        'missed_noorg' : (stats[organ]['missed_noorg']/stats[organ]['total_noorg'])*100,
                        'add_found' : (stats[organ]['additional']/(stats[organ]['total']+stats[organ]['total_noorg']))*100
                        }})
        # All the images have this organ
        else:
            percentages.update({organ : {
                        'found' : (stats[organ]['found']/stats[organ]['total'])*100,
                        'missed' : (stats[organ]['missed']/stats[organ]['total'])*100,
                        'add_found' : (stats[organ]['additional']/stats[organ]['total'])*100
                        }})

    return percentages


def IoU(gt,pred,ratio):
    '''Computes the average IoU for all organs.

    Inputs:
        gt: ground truth annotations.
        pred: predicted annotations.
        ratio: the ratio between the input size and output size.

    Returns:
        all_IoU: dictionnary that contains all IoU for each organs and each files,
                plus the mean and variance for the organs.
                all_IoU = {'mean' : {'bladder' : .., 'rectum' : .., 'prostate' : ..,},
                           'variance': {...},
                           'file1' : {...},
                           'file2': {...},...}
    '''

    # Useful variable
    start = True

    for filename in gt.keys():
        true_boxes = gt[filename]['bb']
        pred_boxes = pred[filename]

        # Initialize the statistics
        if start:
            IoU = {}
            all_IoU = {}
            for o in pred[filename].keys():
                IoU.update({o : []})
            start = False

        # Update filename entry in average IoU
        all_IoU.update({filename : {}})

        for o in pred_boxes.keys():
            if o in true_boxes.keys():
                if len(pred_boxes[o]) > 0:
                    if len(pred_boxes[o]) > 1:
                        # Sort from least confident to most confident
                        pred_boxes[o].sort(key=lambda x:x[4])

                    pred_box_to_compare = [elem*ratio for elem in pred_boxes[o][-1][0:4]]

                    # Computes IoU and updates dict
                    iou_val = compute_IoU(true_boxes[o],pred_box_to_compare)
                    IoU[o].append(iou_val)
                    all_IoU[filename].update({o : iou_val})

    # Updates dictionnary to store average and variance
    all_IoU.update({'mean' : {}, 'variance' : {}})
    for organ in IoU.keys():
        all_IoU['mean'].update({organ : np.mean(IoU[organ])})
        all_IoU['variance'].update({organ : np.std(IoU[organ])})

    return all_IoU

def classification(gt,pred,input_dim,ratio):
    '''Computes the average TP, TN, FP, FN, precision, recall
        and false negative rate for all organs.

    Inputs:
        gt: ground truth annotations.
        pred: predicted annotations.
        ratio: the ratio between the input size and output size.

    Returns:
        classification_stats: a dictionnary that contains all the classification statistics.
                              classification_stats = {'bladder' : {TP : {'mean': .., 'variacnce': ..},
                                                                   TN : {..},
                                                                   FP : {..},
                                                                   FN : {..},
                                                                   P  : {..},
                                                                   R  : {..},
                                                                   FNR: {..}},
                                                        'rectum': {...},...}
    '''

    # Useful variable
    start = True

    for filename in gt.keys():
        true_boxes = gt[filename]['bb']
        pred_boxes = pred[filename]

        # Initialize the statistics
        if start:
            classification = {}
            classification_stats = {}
            for o in pred[filename].keys():
                classification.update({o : {'TP' : [],'TN' : [],'FP' : [],'FN' : [], 'P' : [], 'R' : [], 'FNR' : []}})
                classification_stats.update({o : {'TP' : {'mean' : 0, 'variance' : 0},'TN' : {'mean' : 0, 'variance' : 0},'FP' : {'mean' : 0, 'variance' : 0},'FN' : {'mean' : 0, 'variance' : 0}, 'P' : {'mean' : 0, 'variance' : 0}, 'R' : {'mean' : 0, 'variance' : 0}, 'FNR' : {'mean' : 0, 'variance' : 0}}})
            start = False

        for o in pred_boxes.keys():
            if o in true_boxes.keys():
                if len(pred_boxes[o]) > 0:
                    if len(pred_boxes[o]) > 1:
                        # Sort from least confident to most confident
                        pred_boxes[o].sort(key=lambda x:x[4])

                    # Scale the bounding box according to input/output ratio
                    pred_box_to_compare = [elem*ratio for elem in pred_boxes[o][-1][0:4]]

                    # Computes IoU and updates dict
                    (TP,TN,FP,FN,precision,recall,FNR) = compute_classification(true_boxes[o],pred_box_to_compare,[input_dim[0],input_dim[1]])
                    classification[o]['TP'].append(TP)
                    classification[o]['TN'].append(TN)
                    classification[o]['FP'].append(FP)
                    classification[o]['FN'].append(FN)
                    classification[o]['P'].append(precision)
                    classification[o]['R'].append(recall)
                    classification[o]['FNR'].append(FNR)

    for organ,classes in classification.items():
        for c in classes.keys():
            classification_stats[organ][c].update({ 'mean' : np.mean(classification[organ][c]), 'variance' : np.std(classification[organ][c])})

    return classification_stats

def cases(IoU,tolerance = 0.1,bad_thresh = 0.35):
    '''Takes the dictionnary containing the IoU data and identifies the average and bad_cases.

    Inputs:
        IoU: a dictionnary tht contains the IoU data as returned from the IoU function.
        tolerance: the tolerance for accepting an average case. average+-(var+tolerance) is considered an average case.
        bad_thresh: threshold for considering a bad case. bad case < average-bad_thresh.

    Returns:
        cases: a dictionnary that contains the average and bad cases names.
                cases = {'average': [file1,file2,...], 'bad': [...]}
    '''

    # Storing mean and variance for each organs
    mean = IoU['mean']
    var = IoU['variance']

    # Initializing dictionnary
    cases = {'average' : [], 'bad' : []}

    # Iterating through the dictionnary
    for file,organs in IoU.items():
        # Defining a counter for knowing how many organs fall in the average case
        counter = 0
        goal = len(organs)

        if file != 'mean' and file != 'variance':
            for organ,iou in organs.items():
                if iou <= mean[organ]+(var[organ]+tolerance) and iou >= mean[organ]-(var[organ]+tolerance):
                    counter += 1
                elif file not in cases['bad'] and iou <= mean[organ]-bad_thresh:
                    cases['bad'].append(file)

            if counter == goal:
                # All the organs fall in the average case
                cases['average'].append(file)

    return cases


############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
