'''
Script to use YOLOv2's to compute bounding boxes on custom data.

Adapted from github.com/allanzelener/YAD2K by Luca Derumier.
Version 1.0 - May 2020.
'''

import argparse
import os
import pickle
import numpy as np
import PIL
import matplotlib.pyplot as plt

from utils import save_annotation, load_annotation
from training import normalize, get_classes, process_data, get_detector_mask, create_model, predict
from config import Config

#########################################################
################### Parsing arguments ###################
#########################################################

argparser = argparse.ArgumentParser(
    description="Runs the detection system on custom data.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to the folder containing the data.",
    default=os.path.join('pelvis_scan','data','train'))

argparser.add_argument(
    '-f',
    '--file',
    help="name of the file that contains the ground truth annotations (should be in data_path/(train,val or test)/ folder).",
    default='pelvis_data_train.npz')

argparser.add_argument(
    '-t',
    '--training_path',
    help="path to the folder where all the training data set and related files are stored.",
    default=os.path.join('pelvis_scan','data','train'))

argparser.add_argument(
    '-m',
    '--model_dir',
    help="path to the folder where the model files are stored.",
    default='model_data')

argparser.add_argument(
    '-c',
    '--classes',
    help='name of classes txt file (should be in model_dir).',
    default='pelvis_classes.txt')

argparser.add_argument(
    '-r',
    '--results_dir',
    help="path to the folder where the results are going to be stored.",
    default='results')

argparser.add_argument(
    '-w',
    '--weights',
    help="name of the weights file that we want to load (should be in 'models' directory).",
    default='')

argparser.add_argument(
    '-S',
    '--save',
    help="enables saving the images with bounding box annotations as jpg files.",
    action='store_true')


########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    save = args.save
    data_path = args.data_path
    filename = args.file
    results_dir = args.results_dir
    training_path = args.training_path
    model_dir = args.model_dir
    classes = args.classes
    weights_name = args.weights

    # Computed arguments
    classes_path = os.path.join(model_dir, classes)
    dir_list = [x for x in sorted(os.listdir(data_path)) if x.endswith('.jpg')]


    # Creating config instance
    config = Config()

    # Extracting classes and anchors
    class_names = get_classes(classes_path)
    anchors = config.YOLO_ANCHORS

    # Loading dictionnary
    data = np.load(os.path.join(data_path,filename),allow_pickle=True)

    # Extracting images and boxes
    image_data, boxes = process_data(data['images'], data['boxes'])

    # Extracting anchor boxes and masks
    detectors_mask, matching_true_boxes = get_detector_mask(boxes, anchors)

    # Normalizing data
    normalized_data = normalize(image_data,training_path,train=False)

    # Creating model and printing summary
    model_body, model = create_model(anchors, class_names,freeze_body=config.FREEZE,load_pretrained=config.LOAD_PRETRAINED)

    # Call to predict function
    boxes_dict = predict(model_body,
        class_names,
        anchors,
        normalized_data,
        weights_name,
        dir_list,
        non_best_sup=config.NON_BEST_SUP,
        results_dir=results_dir,
        save=save)

    # Saving predictions
    save_annotation(boxes_dict,os.path.join(results_dir,'predictions','pred_boxes.p'))

#########################################################
################### Utility functions ###################
#########################################################

def non_best_suppression(boxes,classes,scores):
    '''Filters out the boxes, classes and scores that are not associated with the maximum score.

    Inputs:
        boxes: np array of the box coordinates arrays.
        classes: np array of classes associated with the boxes.
        scores: np array of scores associated with the boxes.

    Returns:
        new_box: np array containing the maximum confidence score box.
        new_class: np array containing the maximum confidence score class associated with the box.
        new_score: np array containing the maximum confidence score.
    '''
    # Check consistency in size
    assert len(classes) == len(scores)
    assert len(boxes) == len(scores)

    new_class = []
    new_box = []
    new_score = []
    for idx in range(len(classes)):
        if(classes[idx] not in new_class):
            new_class.append(classes[idx])
            new_box.append(boxes[idx])
            new_score.append(scores[idx])
        elif(new_score[new_class.index(classes[idx])] < scores[idx]):
            swipe_idx = new_class.index(classes[idx])
            new_score[swipe_idx] = scores[idx]
            new_box[swipe_idx] = boxes[idx]

    new_box = np.asarray(new_box)
    new_class = np.asarray(new_class)
    new_score = np.asarray(new_score)

    return (new_box,new_class,new_score)


############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
