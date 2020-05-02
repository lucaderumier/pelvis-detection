'''
Utility functions.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import pickle
import numpy as np

def compute_IoU(box1,box2):
    '''Computes IoU between box1 and box2.
    Inputs:
        box1,2: arrays of length 4 for the box coordinates.

    Returns:
        iou: the intersection over union of box1 and box2.
    '''

    xA = max(box1[0],box2[0]) # Top intersection
    yA = max(box1[1],box2[1]) # Left intersection
    xB = min(box1[2],box2[2]) # Bottom intersection
    yB = min(box1[3],box2[3]) # Right interstion
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    BoxTrueArea = (box1[2]-box1[0])*(box1[3]-box1[1])
    BoxPredArea = (box2[2]-box2[0])*(box2[3]-box2[1])
    assert BoxTrueArea > 0
    assert BoxPredArea > 0
    iou = interArea / (BoxTrueArea + BoxPredArea - interArea)
    return iou

def compute_classification(gt,pred,dim):
    '''Computes TP,TN,FP,FN. dim = (h,w).

    Inputs:
        gt: ground truth bounding box.
        pred: predicted bounding box.
        dim: [h,w] of the full images.

    Returns:
        (TP,TN,FP,FN,precision,recall,FNR): the classification metrics.
    '''

    [xA1,yA1,xB1,yB1] = gt
    [xA2,yA2,xB2,yB2]  = pred
    [h,w] = dim

    xA = max(gt[0],pred[0]) # Top intersection
    yA = max(gt[1],pred[1]) # Left intersection
    xB = min(gt[2],pred[2]) # Bottom intersection
    yB = min(gt[3],pred[3]) # Right interstion

    # Useful areas
    totalArea = h*w
    interArea = max(0.0, xB - xA) * max(0.0, yB - yA)
    gtArea = (xB1-xA1)*(yB1-yA1)
    predArea = (xB2-xA2)*(yB2-yA2)

    assert totalArea >= 0
    assert interArea >= 0
    assert gtArea >= 0
    assert predArea >= 0

    # Classification
    TP = interArea
    TN = (totalArea-gtArea-predArea+interArea)
    FP = (predArea-interArea)
    FN = (gtArea-interArea)

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    FNR = FN/(TP+FN)

    return (TP,TN,FP,FN,precision,recall,FNR)

def save_annotation(dict,annot_path):
    '''Saves annot_file into a dictionnary.

    Inputs:
        dict: the dictionnary to be saved.
        annot_path: the path to the file where it will be saved.
    '''

    with open(annot_path, 'wb') as fp:
        pickle.dump(dict, fp)

def load_annotation(annot_path):
    '''Loads annot_file into a dictionnary.

    Inputs:
        annot_path: the path to the file that will be loaded.

    Returns:
        data: the dictionnary containing the data from annpt_path.
    '''

    with open(annot_path, 'rb') as fp:
        data = pickle.load(fp)

    return data
