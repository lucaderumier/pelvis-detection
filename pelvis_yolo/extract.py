'''
Script to extract 3D bounding boxes from the 2D slices.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import argparse
import os
import numpy as np

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
    help="path to the folder containing the data folders.",
    default=os.path.join('pelvis_scan','FULL_IMAGES_CT'))

argparser.add_argument(
    '-i',
    '--image',
    help="name of the folder that contains all the 2D slices and ground truth annotations.",
    default='charleroi_1')

argparser.add_argument(
    '-p',
    '--pred',
    help="path to the prediction file.",
    default=os.path.join('predictions','pred_boxes.p'))

########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    data_path = args.data_path
    image_folder = args.image
    pred_path = args.pred

    # Config instance and scaling ratio
    config = Config()
    ratio = (config.INPUT_DIM[0]/config.OUTPUT_DIM[0])

    # Checking good composition of the image folders
    contentChecker(data_path,pred_path)

    # Extracting results for 3D boxes
    metrics = extract_all(data_path,pred_path,ratio=ratio)

    # Saving as p file
    save_annotation(metrics,os.path.join(data_path,'extract_stats.p'))

    # Writing directories to a file
    f = open(os.path.join(data_path,'extract_stats.txt'),'w+')
    for key,item in metrics.items():
        f.write('{} : {}\n\n'.format(key,item))
    f.close()

############################################################
################### Extraction functions ###################
############################################################
def extract_all(path,pred_path,ratio=500/416,input_dim=[500,500]):
    '''Extract 3D boxes and computes IoU compared to ground truth of all the images contained in path.

    Inputs:
        path: the path to the images folders.
        pred_path: the path to the predictions file that has to be stored in the image folder.

    Returns:
        metrics: the dictionnary containing the mean and variance of the metrics for 3D boxes.

    '''

    dir_list = [x for x in os.listdir(path) if x.startswith('charleroi')]

    IoUs = []
    areas = []
    TPs = []
    TNs = []
    FPs = []
    FNs = []
    precs = []
    recs = []
    FNRs = []


    # Computation of all IoUs and areas phase
    for dir in dir_list:
        # Load prediction and ground truth annotations
        pred_boxes = load_annotation(os.path.join(path,dir,pred_path))
        true_boxes = load_annotation(os.path.join(path,dir,'boxes.p'))
        gt_box = true_boxes['all']

        # Computes total 3D box for predicted boxes
        pred_box_dilated = extract(merge_boxes(pred_boxes))
        pred_box = [x*ratio for x in pred_box_dilated]

        # Computes and stores IoU and area of the box
        IoU = compute_IoU(pred_box,gt_box)
        IoUs.append(IoU)

        (TP,TN,FP,FN,precision,recall,FNR) = compute_classification(gt_box,pred_box,input_dim)
        TPs.append(TP)
        TNs.append(TN)
        FPs.append(FP)
        FNs.append(FN)
        precs.append(precision)
        recs.append(recall)
        FNRs.append(FNR)

        area = (pred_box[2]-pred_box[0])*(pred_box[3]-pred_box[1]) #(xB-xA)*(yB-yA)
        areas.append(area)

    # Extracting mean and variance
    final_iou = {'mean': np.mean(IoUs), 'variance': np.var(IoUs)}
    final_area = {'mean': np.mean(areas), 'variance': np.var(areas)}
    final_TP = {'mean': np.mean(TPs), 'variance': np.var(TPs)}
    final_TN = {'mean': np.mean(TNs), 'variance': np.var(TNs)}
    final_FP = {'mean': np.mean(FPs), 'variance': np.var(FPs)}
    final_FN = {'mean': np.mean(FNs), 'variance': np.var(FNs)}
    final_precision = {'mean': np.mean(precs), 'variance': np.var(precs)}
    final_recall = {'mean': np.mean(recs), 'variance': np.var(recs)}
    final_FNR = {'mean': np.mean(FNRs), 'variance': np.var(FNRs)}

    metrics = {'IoU' : final_iou, 'area' : final_area, 'TP' : final_TP, 'TN' : final_TN, 'FP': final_FP, 'precision' : final_precision, 'recall' : final_recall, 'FNR' : final_FNR}

    return metrics

def merge_boxes(pred_boxes):
    '''Merge the organ boxes into a dictionnary.

    Inputs:
        pred_boxes: the dictionnary containing the bounding boxes and scores.
                    pred_boxes = {filename : {'bladder': [[xA,yA,xB,yB,score],[...]],
                                              'rectum': [..],
                                              'prostate': [..]},
                                  filename : {...},...}

    Returns:
        total_boxes: the dictionnary containing the bounding boxes that and scores that contain all the organs of each slide.
                     total_boxes = {filename1 : [xA,yA,xB,yB,score],
                                    filename2 : [xA,yA,xB,yB,score],...}

    '''

    total_boxes = {}
    for filename,organs in pred_boxes.items():
        # New coordinates
        xA = 10000
        yA = 10000
        xB = -1
        yB = -1
        conf = 1

        # Extract the
        for organ,boxes in organs.items():
            if len(boxes) > 0:

                # Sort the boxes if there is more than 1 prediction
                if len(boxes) > 1:
                    # Sort from least confident to most confident
                    boxes.sort(key=lambda x:x[4])

                # Keep only box that has the best confidence score if this score is above 0.5
                box = boxes[-1]
                if(box[-1] > 0.5):
                    xA = min(xA,box[0])
                    yA = min(yA,box[1])
                    xB = max(xB,box[2])
                    yB = max(yB,box[3])
                    conf = min(conf,box[4])

        # Adds the bounding box to the new dictionnary if it has been update
        if(xA != 10000 and yA != 10000 and xB > 0 and yB > 0):
            total_boxes.update({filename : [xA,yA,xB,yB,conf]})

    return total_boxes

def extract(total_boxes):
    '''Extract one bounding box that holds every others inside it.

    Inputs:
        total_boxes: the dictionnary containing the bounding boxes that and scores that contain all the organs of each slide.
                     total_boxes = {filename1 : [xA,yA,xB,yB,score],
                                    filename2 : [xA,yA,xB,yB,score],...}

    Returns:
        [xA,yA,xB,yB]: the final box coordinates for the 3D image

    '''

    xA = 10000
    yA = 10000
    xB = -1
    yB = -1

    for filename,box in total_boxes.items():
        # New coordinates
        xA = min(xA,box[0])
        yA = min(yA,box[1])
        xB = max(xB,box[2])
        yB = max(yB,box[3])

    if(xA >= 10000 or yA >= 10000 or xB < 0 or yB < 0):
        raise ValueError('Some coordinates were never (or wrongly) updated during the final box extraction.')

    return [xA,yA,xB,yB]

#########################################################
################### Utility functions ###################
#########################################################
def contentChecker(path,pred_path):
    '''Checks that all image folders in path have predictions and ground truth data.

    Inputs:
        path: the path to the images folders.
        pred_path: the path to the predictions file that has to be stored in the image folder.

    '''

    dir_list = [x for x in os.listdir(path) if x.startswith('charleroi')]

    # Computation of all IoUs and areas phase
    for dir in dir_list:
        if(not os.path.exists(os.path.join(path,dir,pred_path))):
            print('Missing predictions for {}'.format(dir))
        if(not os.path.exists(os.path.join(path,dir,'boxes.p'))):
            print('Missing ground truth for {}'.format(dir))


############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
