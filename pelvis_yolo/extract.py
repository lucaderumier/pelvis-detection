'''
Script to extract 3D bounding boxes from the 2D slices.

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

argparser.add_argument(
    '-s',
    '--scale',
    help="scaling factor for the bounding box.",
    default='0')

argparser.add_argument(
    '-l',
    '--limit',
    help="scaling factor limit for the bounding box.",
    default='20')

argparser.add_argument(
    '-G',
    '--graphs',
    help="enables graph mode.",
    action='store_true')

argparser.add_argument(
    '-E',
    '--each',
    help="enables independent organ extraction mode.",
    action='store_true')



########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    data_path = args.data_path
    image_folder = args.image
    pred_path = args.pred
    scaling = int(args.scale)
    limit = int(args.limit)
    g_mode = args.graphs
    each = args.each

    # Config instance and scaling ratio
    config = Config()
    ratio = (config.INPUT_DIM[0]/config.OUTPUT_DIM[0])

    # Checking good composition of the image folders
    contentChecker(data_path,pred_path)

    # Computes average box
    #save_annotation(averageBox('/Volumes/LUCA_EHD/TFE_DATA/USABLE_DATASETS/FULL_IMAGES_CT',[5,18,27,35,45,59,61,62,65,71,75,89,91]),os.path.join(data_path,'average_box_training.p'))
    #average_box = [int(x) for x in load_annotation(os.path.join(data_path,'average_box_training.p'))]
    #average_comparison(average_box,data_path,pred_path,ratio=ratio,scaling = 0,write=True)

    # Study scaling impact
    if g_mode:
        metrics_graphs(data_path,os.path.join(data_path,'extract_scaling.p'))
    elif each:
        metrics = extract_each(data_path,pred_path,ratio=ratio,write=True)
    else:
        metrics = extract_scaling(data_path,pred_path,ratio=ratio,limit=limit,write=True)
        save_annotation(metrics,os.path.join(data_path,'extract_scaling.p'))


############################################################
################### Extraction functions ###################
############################################################
def average_comparison(average_box,path,pred_path,ratio=500/416,scaling = 0,write=False):
    '''Computes the average IoU between the average training gt box and test gt box against predictions.

    Inputs:
        average_box: the average box position from training set.
        path: the path to the images folders.
        pred_path: the path to the predictions file that has to be stored in the image folder.

    Returns:
        IoUs: the list containing IoU of ground truth and predictions.
    '''
    GTs = []
    preds = []
    IoUs = []
    dir_list = [x for x in os.listdir(path) if x.startswith('charleroi')]
    for dir in dir_list:

        # Load prediction and ground truth annotations
        pred_boxes = load_annotation(os.path.join(path,dir,pred_path))
        true_boxes = load_annotation(os.path.join(path,dir,'boxes.p'))
        gt_box = true_boxes['all']

        # Computes total 3D box for predicted boxes
        pred_box_dilated = extract(merge_boxes(pred_boxes))
        pred_box = [x*ratio for x in pred_box_dilated]

        if scaling > 0:
            pred_box = [pred_box[0]-scaling, pred_box[1]-scaling, pred_box[2]+scaling, pred_box[3]+scaling]

        # Computes and stores IoU and area of the box
        GTs.append(compute_IoU(average_box,gt_box))
        preds.append(compute_IoU(average_box,pred_box))
        IoUs.append(compute_IoU(pred_box,gt_box))

    if write:
        f = open(os.path.join(path,'compare.txt'),'w+')
        for i in range(len(GTs)):
            f.write('patient {} : {:.3f} | {:.3f} | {:.3f}\n\n'.format(i,GTs[i],preds[i],IoUs[i]))
        f.write('average : {} | {} | {}\n\n'.format(np.mean(GTs),np.mean(preds),np.mean(IoUs)))
        f.write('std : {} | {} | {}\n\n'.format(np.std(GTs),np.std(preds),np.std(IoUs)))
        f.close()

    return [GTs,preds,IoUs]

def extract_scaling(path,pred_path,ratio=500/416,input_dim=[500,500],limit=20,write=False):
    '''Extract 3D boxes and computes all metrics for scaling factor in the range [0;limit]

    Inputs:
        path: the path to the images folders.
        pred_path: the path to the predictions file that has to be stored in the image folder.

    Returns:
        metrics: the dictionnary containing the mean and variance of the metrics for 3D boxes for each scaling factor.
    '''

    metrics = {}
    for s in range(0,limit+1):
        m = extract_all(path,pred_path,ratio=ratio,input_dim=input_dim,scaling=s)
        metrics.update({s : m})

        if write:
            f = open(os.path.join(path,'extract_stats_'+str(s)+'.txt'),'w+')
            for key,item in m.items():
                f.write('{} : {}\n\n'.format(key,item))
            f.close()

    return metrics

def extract_all(path,pred_path,ratio=500/416,input_dim=[500,500],scaling=0):
    '''Extract 3D boxes and computes IoU compared to ground truth of all the images contained in path along with other metrics.

    Inputs:
        path: the path to the images folders.
        pred_path: the path to the predictions file that has to be stored in the image folder.

    Returns:
        metrics: the dictionnary containing the mean and variance of the metrics for 3D boxes.

    '''

    dir_list = [x for x in os.listdir(path) if x.startswith('charleroi')]

    IoUs = []
    areas = []
    normal_area = []
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
        if scaling > 0:
            pred_box = [pred_box[0]-scaling, pred_box[1]-scaling, pred_box[2]+scaling, pred_box[3]+scaling]

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
        normal_area.append(area/(input_dim[0]*input_dim[1]))



    # Extracting mean and variance
    final_iou = {'mean': np.mean(IoUs), 'variance': np.std(IoUs)}
    final_area = {'mean': np.mean(areas), 'variance': np.std(areas)}
    final_normal_area = {'mean': np.mean(normal_area), 'variance': np.std(normal_area)}
    final_TP = {'mean': np.mean(TPs), 'variance': np.std(TPs)}
    final_TN = {'mean': np.mean(TNs), 'variance': np.std(TNs)}
    final_FP = {'mean': np.mean(FPs), 'variance': np.std(FPs)}
    final_FN = {'mean': np.mean(FNs), 'variance': np.std(FNs)}
    final_precision = {'mean': np.mean(precs), 'variance': np.std(precs)}
    final_recall = {'mean': np.mean(recs), 'variance': np.std(recs)}
    final_FNR = {'mean': np.mean(FNRs), 'variance': np.std(FNRs)}

    metrics = {'IoU' : final_iou, 'area' : final_area, 'norm_area' : final_normal_area, 'TP' : final_TP, 'TN' : final_TN, 'FP': final_FP, 'FN': final_FN, 'precision' : final_precision, 'recall' : final_recall, 'FNR' : final_FNR}

    return metrics

def extract_each(path,pred_path,ratio=500/416,input_dim=[500,500],scaling=0,write=False):
    '''Extract 3D boxes and computes IoU compared to ground truth of all the images (for each organ)) contained in path along with other metrics.

    Inputs:
        path: the path to the images folders.
        pred_path: the path to the predictions file that has to be stored in the image folder.

    Returns:
        metrics: the dictionnary containing the mean and variance of the metrics for 3D boxes.

    '''

    dir_list = [x for x in os.listdir(path) if x.startswith('charleroi')]

    metrics = {'bladder' : {'IoU' : [], 'precision' : [], 'recall' : [], 'FNR' : []},
                'rectum' : {'IoU' : [], 'precision' : [], 'recall' : [], 'FNR' : []},
                'prostate' : {'IoU' : [], 'precision' : [], 'recall' : [], 'FNR' : []},
                'average' : {'IoU' : [], 'precision' : [], 'recall' : [], 'FNR' : []}}


    # Computation of all IoUs and areas phase
    for dir in dir_list:

        # Load prediction and ground truth annotations
        pred_boxes = merge_each(load_annotation(os.path.join(path,dir,pred_path)))
        true_boxes = merge_each(format(load_annotation(os.path.join(path,dir,'boxes.p'))))

        for organ,box in pred_boxes.items():
            pred_box = [x*ratio for x in box][:4]

            if scaling > 0:
                pred_box = [pred_box[0]-scaling, pred_box[1]-scaling, pred_box[2]+scaling, pred_box[3]+scaling]

            IoU = compute_IoU(pred_box,true_boxes[organ][:4])
            metrics[organ]['IoU'].append(IoU)
            metrics['average']['IoU'].append(IoU)
            (TP,TN,FP,FN,precision,recall,FNR) = compute_classification(true_boxes[organ][:4],pred_box,input_dim)
            metrics[organ]['precision'].append(precision)
            metrics['average']['precision'].append(precision)
            metrics[organ]['recall'].append(recall)
            metrics['average']['recall'].append(recall)
            metrics[organ]['FNR'].append(FNR)
            metrics['average']['FNR'].append(FNR)

    if write:
        f = open(os.path.join(path,'organ_stats.txt'),'w+')
        f.write('organ - iou - precision - recall - fnr\n\n')
        for key,item in metrics.items():
            f.write('{} : {:.3f}+{:.3f} | {:.3f}+{:.3f} | {:.3f}+{:.3f} | {:.3f}+{:.3f}\n\n'.format(key,np.mean(item['IoU']),np.std(item['IoU']),np.mean(item['precision']),np.std(item['precision']),np.mean(item['recall']),np.std(item['recall']),np.mean(item['FNR']),np.std(item['FNR'])))
        f.close()

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

def merge_each(pred_boxes):
    '''Merge the organ boxes into a dictionnary by organ.

    Inputs:
        pred_boxes: the dictionnary containing the bounding boxes and scores.
                    pred_boxes = {filename : {'bladder': [[xA,yA,xB,yB,score],[...]],
                                              'rectum': [..],
                                              'prostate': [..]},
                                  filename : {...},...}

    Returns:
        total_boxes: the dictionnary containing the bounding boxes that and scores that contain all the organs of each slide.
                     organ_boxes = {'bladder' : [xA,yA,xB,yB,score], 'rectum' :[xA,yA,xB,yB,score], 'prostate' : [xA,yA,xB,yB,score]}

    '''
    # New coordinates
    xA = 10000
    yA = 10000
    xB = -1
    yB = -1
    conf = 1

    organ_boxes = {'bladder': [xA,yA,xB,yB,conf],'rectum': [xA,yA,xB,yB,conf],'prostate': [xA,yA,xB,yB,conf]}
    for filename,organs in pred_boxes.items():
        # Extract the boxes
        for organ,boxes in organs.items():
            if len(boxes) > 0:
                # Sort the boxes if there is more than 1 prediction
                if len(boxes) > 1:
                    # Sort from least confident to most confident
                    boxes.sort(key=lambda x:x[4])

                # Keep only box that has the best confidence score if this score is above 0.5
                box = boxes[-1]
                if(box[-1] > 0.5):
                    organ_boxes[organ][0] = min(organ_boxes[organ][0],box[0]) # xA
                    organ_boxes[organ][1] = min(organ_boxes[organ][1],box[1]) # yA
                    organ_boxes[organ][2] = max(organ_boxes[organ][2],box[2]) # xB
                    organ_boxes[organ][3] = max(organ_boxes[organ][3],box[3]) # yB
                    organ_boxes[organ][4] = min(organ_boxes[organ][4],box[4]) # conf


    return organ_boxes

def format(true_boxes):
    '''Transform the true boxes dictionnary to have a pred boxes format.

    Inputs:
        true_boxes: the dictionnary containing the ground truth bounding boxes.
                     total_boxes = {filename1 : {'bb' : {'bladder' : [xA,yA,xB,yB], 'prostate' : [], ..}, 'shape': (....)},
                                    filename2 : {'bb' : {'bladder' : [xA,yA,xB,yB], 'prostate' : [], ..}, 'shape': (....)}}

    Returns:
        boxes: the reformated dictionnary
                    boxes = {filename1 : {'bladder': [[xA,yA,xB,yB,score]],
                                            'rectum': [..],
                                            'prostate': [..]},
                                  filename : {...},...}
    '''
    boxes = {}
    for file in true_boxes.keys():
        boxes.update({file : {'bladder' : [], 'rectum': [], 'prostate':[]}})
        if type(true_boxes[file]) is dict :
            for organ,box in true_boxes[file]['bb'].items():

                if organ in boxes[file].keys():
                    boxes[file][organ].append(box + [0.99])
    return boxes


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

def averageBox(path,ignore):
    '''Extract the average box of the training set.

    Inputs:
        path: path to the folder that contains the images.
        ignore: numbers to ignore (data from validation and test set).

    Returns:
        [xA,yA,xB,yB]: the final box coordinates for the 3D image.

    '''

    dir_list = [x for x in os.listdir(path) if x.startswith('charleroi')]
    xAs = []
    yAs = []
    xBs = []
    yBs = []
    for file in dir_list:
        if not (int(re.findall('\d+', file)[0]) in ignore):
            box = load_annotation(os.path.join(path,file,'boxes.p'))
            xAs.append(box['all'][0])
            yAs.append(box['all'][1])
            xBs.append(box['all'][2])
            yBs.append(box['all'][3])

    return [np.mean(xAs),np.mean(yAs),np.mean(xBs),np.mean(yBs)]

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

def metrics_graphs(path,extract_path):
    '''Plots graphs of the metrics in terms of scaling factor.

    Inputs:
        path: the path to the images folders.
        pred_path: the path to the file that stores the extraction results.

    '''

    metrics = load_annotation(extract_path)
    iou = np.zeros((2,len(metrics)))
    precision = np.zeros((2,len(metrics)))
    recall = np.zeros((2,len(metrics)))
    fnr = np.zeros((2,len(metrics)))
    area = np.zeros((2,len(metrics)))

    for s,m in metrics.items():
        iou[0][s] = m['IoU']['mean']
        iou[1][s] = m['IoU']['variance']
        precision[0][s] = m['precision']['mean']
        precision[1][s] = m['precision']['variance']
        recall[0][s] = m['recall']['mean']
        recall[1][s] = m['recall']['variance']
        fnr[0][s] = m['FNR']['mean']
        fnr[1][s] = m['FNR']['variance']
        area[0][s] = m['norm_area']['mean']
        area[1][s] = m['norm_area']['variance']

        if(m['FNR']['mean'] == 0.0):
            max = s
            break

    data_metrics = [iou[0][0:s+1],precision[0][0:s+1],recall[0][0:s+1],fnr[0][0:s+1]]
    var_metrics = [iou[1][0:s+1],precision[1][0:s+1],recall[1][0:s+1],fnr[1][0:s+1]]
    extract_graph(data_metrics,var_metrics,list(range(s+1)),['IoU','precision','recall','fnr'],['mediumpurple','darkseagreen','darkorange','tomato'],ylabel='',title='',save=True,path=os.path.join(path,'extract_metrics.pdf'))
    extract_graph([area[0][0:s+1]],[area[1][0:s+1]],list(range(s+1)),['volume ratio'],['royalblue'],ylabel='volume ratio',title='',save=True,path=os.path.join(path,'extract_GPU.pdf'))
    #av_64 = [x*578 for x in area[0][0:s+1]]
    im = [(x*578)/8 for x in area[0][0:s+1]]
    mask = [(x*1729)/8 for x in area[0][0:s+1]]
    #max = [x*1058 for x in area[0][0:s+1]]
    #min = [x*431 for x in area[0][0:s+1]]
    extract_graph([im,mask],[np.zeros((s+1)),np.zeros((s+1))],list(range(s+1)),['image','mask'],['royalblue','red'],ylabel='data size in MB',title='',save=True,path=os.path.join(path,'extract_size_images.pdf'),mem_lines=True)
    #av_64 = [x*1729 for x in area[0][0:s+1]]
    #av_8 = [(x*1729)/8 for x in area[0][0:s+1]]
    #max = [x*3175 for x in area[0][0:s+1]]
    #min = [x*1293 for x in area[0][0:s+1]]
    #extract_graph([av_64,av_8],[np.zeros((s+1)),np.zeros((s+1))],list(range(s+1)),['uint64','uint8'],['steelblue','darkseagreen'],ylabel='data size in MB',title='',save=True,path=os.path.join(path,'extract_size_mask.pdf'))



############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
