'''
Script to extract 3D bounding boxes from the 2D slices.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import argparse
import os

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
    default=os.path.join('results','predictions','pred_boxes.p'))

########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    data_path = args.data_path
    image_folder = args.image
    pred_path = args.pred

    # Boxes annotations
    pred_boxes = load_annotation(pred_path)
    true_boxes = load_annotation(os.path.join(data_path,image_folder,'boxes.p'))

    # Config instance and scaling ratio
    config = Config()
    ratio = (config.INPUT_DIM[0]/config.OUTPUT_DIM[0])

    # Ground truth and predicted 3D box coordinates
    gt_box = true_boxes['all']
    pred_box_dilated = extract(merge_boxes(pred_boxes))
    pred_box = [x*ratio for x in pred_box_dilated]
    print(compute_IoU(pred_box,gt_box))


############################################################
################### Extraction functions ###################
############################################################
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

############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
