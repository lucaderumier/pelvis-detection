'''
Utility functions.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import os
import pickle
import numpy as np
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw, ImageEnhance
from config import Config

####################################################
################# Global variables #################
####################################################

config = Config()

####################################################
################### Computations ###################
####################################################

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

def transform_box(box,ratio):
    '''Takes bounding box and rescales it according to ratio.

    Inputs:
        box: the box coordinates as a list.
        ratio: the rescaling ratio

    Returns:
        rescaled_box: the rescaled box.
    '''

    return [elem*ratio for elem in box]

def nan_to_x(number,x=0):
    '''Returns the number or 0 if it's nan.

    Inputs:
        number: float or nan.

    Retruns:
        x: number or x if number is nan.
    '''

    if(np.isnan(number)):
        return x
    return number

#####################################################
################### Visualization ###################
#####################################################

def draw_bb(im_path,bb,to_draw=['bladder','rectum','prostate'],colors=['darkgreen','red','blue','yellow'],save=False,results_path='',rescale=False,ratio=500/416):
    '''Draws boudning boxes on image.

    Inputs:
        im_path: the path to the image
        bb: the bounding box coordinates.
        to_draw: a list of strings that contains the organs to draw the bounding box for.
        colors: the colors of the bounding boxes.
        save: wether to save the image or not.
        results_path: the path of the folder where to save the results.
        rescale: wether to rescale the box or not.
        ratio: the rescaling ratio.
    '''

    if(not im_path.endswith('.jpg')):
        raise NameError('File is not a jpeg file.')

    im = Image.open(im_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    i = 0
    for organ in to_draw:
        if(organ in bb.keys() and bb[organ] is not None and len(bb[organ]) != 0):
            if rescale:
                bb[organ].sort(key=lambda x:x[4])
                box = transform_box(bb[organ][0],ratio)
            else:
                box = bb[organ]

            draw.rectangle([box[1],box[0],box[3],box[2]],outline=colors[i])
        i += 1


    # Extracts the image name
    name = os.path.basename(im_path)

    # Image brightness enhancer
    enhancer = ImageEnhance.Contrast(im)
    factor = 2.5
    im_output = enhancer.enhance(factor)

    if save:
        if rescale:
            im_output.save(os.path.join(results_path,name.replace('.jpg','-pred.jpg')),quality=90)
        else:
            im_output.save(os.path.join(results_path,name.replace('.jpg','-bb.jpg')),quality=90)
    else:
        im.show()

    del draw

def over_draw(im_path,bb_gt,bb_pred,to_draw=['bladder','rectum','prostate'],colors_gt=['darkgreen','red','blue','yellow'],colors_pred=['lime','deeppink','cyan','gold'],save=False,results_path='',ratio=500/416):
    '''Draws both prediction and ground truth bounding boxes on image.

    Inputs:
        im_path: the path to the image
        bb_gt: the ground truth bounding box coordinates.
        bb_pred: the prediction bounding box coordinates.
        to_draw: a list of strings that contains the organs to draw the bounding box for.
        colors_gt: the colors of the ground truth bounding boxes.
        colors_pred: the colors of the prediction bounding boxes.
        save: wether to save the image or not.
        results_path: the path of the folder where to save the results.
        ratio: the rescaling ratio.
    '''

    if(not im_path.endswith('.jpg')):
        raise NameError('File is not a jpeg file.')

    im = Image.open(im_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    i = 0
    for organ in to_draw:
        if(organ in bb_gt.keys() and bb_gt[organ] is not None):
            box_gt = bb_gt[organ]
            draw.rectangle([box_gt[1],box_gt[0],box_gt[3],box_gt[2]],outline=colors_gt[i])

        if(organ in bb_pred.keys() and len(bb_pred[organ]) != 0):
            bb_pred[organ].sort(key=lambda x:x[4])
            box_pred = transform_box(bb_pred[organ][0],ratio)
            draw.rectangle([box_pred[1],box_pred[0],box_pred[3],box_pred[2]],outline=colors_pred[i])
        i += 1

    # Extracts the image name
    name = os.path.basename(im_path)

    # Image brightness enhancer
    enhancer = ImageEnhance.Contrast(im)
    factor = 2.5
    im_output = enhancer.enhance(factor)

    if save:
        im_output.save(os.path.join(results_path,name.replace('.jpg','-over.jpg')),quality=90)
        #im.save(os.path.join(results_path,name.replace('.jpg','-over.jpg')),quality=90)
    else:
        im.show()

    del draw

##############################################
################### Graphs ###################
##############################################

def learning_graph(history,metrics,legend,save=False,path='history.png',scale='linear'):
    '''Plots the learning graph's metrics.

    Inputs:
        history: the dictionnary that contains the history data.
        metrics: the metrcis we want to plot.
        legend: the legend of the plot.
        save: wether to save the plot or not.
        path: the path to the file to be saved.
        scale: the scale of the y values.
    '''

    for m in metrics:
        plt.plot(history[m])

    plt.title('Learning graph')
    plt.ylabel('loss')
    plt.yscale(scale)
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper right')
    if save:
        plt.savefig(path)
    else:
        plt.show()

    plt.close()

def generic_graph(data,var,epochs,legend,ylabel='IoU',title='IoU',save=False,path='iou.png'):
    '''Plots some data metric.

    Inputs:
        data: list of data.
        epochs: list of corresponding epochs
        legend: the legend of the plot.
        ylabel: y axis label.
        title: title of the graph.
        save: wether to save the plot or not.
        path: the path to the file to be saved.
    '''
    for i in range(len(data)):
        plt.errorbar(epochs,data[i],var[i],label = legend[i],capsize=1,linewidth=0.7, elinewidth=0.5,marker='.')

    plt.title(title)
    plt.ylabel(ylabel)
    plt.xlabel('epoch')
    plt.legend(legend)
    if save:
        plt.savefig(path,dpi=1200)
    else:
        plt.show()

    plt.close()


#############################################
################### Files ###################
#############################################

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

def merge_history(path):
    '''Merges all the history files from path into one dictionnary.

    Inputs:
        path: the path to the folder where the history files are stored.

    Returns:
        final_h: the merged history dictionnary.
    '''
    hist_files = [x for x in os.listdir(path) if x.endswith('.p')]

    if len(hist_files) == 0:
        raise ValueError('No history files to merge')

    final_h = {}
    for n in range(len(hist_files)):
        if(os.path.exists(os.path.join(path,'history'+str(n)+'.p'))):
            h = os.path.join(path,'history'+str(n)+'.p')
            hx = load_annotation(h)
            if(n == 0):
                final_h = hx
            else:
                for key,value in hx.items():
                    final_h[key].extend(value)

    return final_h
