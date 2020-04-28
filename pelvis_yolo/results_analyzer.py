import os
import numpy as np
import matplotlib.pyplot as plt
import imageio
import shutil
import pickle
from skimage.color import gray2rgb
from PIL import Image, ImageDraw

# mode
MODE = 'extract' #'train','test_on_train', 'test_on_val' or 'extract'

# files
if(MODE == 'extract'):
    DATA_SET = os.path.join('EXTRACT_TEST') # CHANGE THIS
else:
    DATA_SET = 'DATA_CT'

PREDICTION_FILE = 'pred_boxes.p'
STATS_FILE = 'CT_stats.txt'
IOU_FILE = 'iou.p'
CASE_FILE = 'case.txt'

# directories
ORIGINAL_DATA_DIR = os.path.join('pelvis_scan',DATA_SET)
RESULTS_DIR  = 'RESULTS'
IMAGES_DIR = os.path.join(RESULTS_DIR,'images')
HISTORY_DIR = os.path.join(RESULTS_DIR,'history')

# mode dependent variables
if MODE == 'test':
    ANNOTATION_FILE = 'annotations_test.p'
    TEST_DATA_DIR = os.path.join(ORIGINAL_DATA_DIR,'test')
    NUM_IMAGES = 100
elif MODE == 'test_on_train':
    ANNOTATION_FILE = 'annotations_train.p'
    TEST_DATA_DIR = os.path.join(ORIGINAL_DATA_DIR,'train')
    NUM_IMAGES = 1000
elif MODE == 'test_on_val':
    ANNOTATION_FILE = 'annotations_val.p'
    TEST_DATA_DIR = os.path.join(ORIGINAL_DATA_DIR,'val')
    NUM_IMAGES = 100
elif MODE == 'extract':
    # CHANGE THIS SECTION
    ANNOTATION_FILE = 'boxes.p'
    TEST_DATA_DIR = os.path.join(ORIGINAL_DATA_DIR,'charleroi_1')
    NUM_IMAGES = 92

# Variables
OOI = ['bladder','rectum','prostate']
YOLO_DIM = 416
CT_DIM = 500

def _main():
    # Renames the test result images
    
    copy_images(TEST_DATA_DIR,IMAGES_DIR)
    rename(IMAGES_DIR,PREDICTION_FILE)

    # Draws original bounding boxes
    to_draw = OOI
    dict = load_annotation(os.path.join(IMAGES_DIR,ANNOTATION_FILE))
    save_all(IMAGES_DIR,dict,to_draw)



    # Analyze results.txt
    analyze_results(os.path.join(RESULTS_DIR,'results.txt'),os.path.join(RESULTS_DIR,STATS_FILE),NUM_IMAGES,ooi = OOI)
    average_IoU(IMAGES_DIR,PREDICTION_FILE,ANNOTATION_FILE,os.path.join(RESULTS_DIR,STATS_FILE),os.path.join(RESULTS_DIR,IOU_FILE),OOI,CT_DIM,YOLO_DIM)
    average_classification(IMAGES_DIR,PREDICTION_FILE,ANNOTATION_FILE,os.path.join(RESULTS_DIR,STATS_FILE),OOI,CT_DIM,YOLO_DIM)

    # cases
    case(os.path.join(RESULTS_DIR,IOU_FILE),OOI,os.path.join(RESULTS_DIR,CASE_FILE),tolerance = 0.01)

    '''
    # Analyze history
    hist = concatenate_history(HISTORY_DIR)
    #metrics = ['loss','val_loss','classification_loss','val_classification_loss','coord_loss','val_coord_loss','conf_loss','val_conf_loss']
    #legend = ['training loss (total)', 'validation loss (total)', 'training classification loss','validation classification loss','training coordinates loss','validation coordinates loss','training confidence loss','validation confidence loss']
    metrics = ['coord_loss','val_coord_loss']
    legend = ['training coordinates loss','validation coordinates loss']
    learning_graph(hist,metrics,legend,scale = 'log')
    '''


def save_annotation(dict,annot_path):
    'Save annot_file into a dictionnary.'

    with open(annot_path, 'wb') as fp:
        pickle.dump(dict, fp)

def load_annotation(annot_path):
    '''Loads annot_file into a dictionnary.'''

    with open(annot_path, 'rb') as fp:
        data = pickle.load(fp)

    return data

def copy_images(src_path,dest_path):
    '''Copies all file .p and .jpg from src to dest.'''
    dir_list = sorted(os.listdir(src_path))
    for filename in dir_list:
        if(filename.endswith('.jpg') or filename.endswith('.p')):
            shutil.copy(os.path.join(src_path,filename),os.path.join(dest_path,filename))


def rename(path_data,pred_file):
    '''Renames the png images accordingly to their corresponding jpg images.'''
    dir_list = sorted(os.listdir(path_data))
    pred_dict = load_annotation(os.path.join(path_data,pred_file))
    new_dict = {}

    i = 0
    for filename in dir_list:
        if(filename.endswith('.jpg')):
            new_dict.update({filename : pred_dict[i]})
            png_file = os.path.join(path_data,str(i)+'.png')
            new_png_file = os.path.join(path_data,filename.replace('.jpg','-pred.png'))
            os.rename(png_file,new_png_file)
            i+=1

    save_annotation(new_dict,os.path.join(path_data,pred_file))

def draw_single(im_path,box):
    '''Draws a single box on an image.'''
    im = Image.open(im_path).convert("RGB")
    draw = ImageDraw.Draw(im)
    draw.rectangle([box[1],box[0],box[3],box[2]],outline='green')
    im.show()
    del draw

def draw_bb(im_path,bb,to_draw=['bladder','rectum','prostate'],save=False):
    '''Draws boudning boxes on image.'''

    if(not im_path.endswith('.jpg')):
        raise NameError('File is not a jpeg file.')

    im = Image.open(im_path).convert("RGB")
    draw = ImageDraw.Draw(im)

    if('bladder' in bb.keys() and 'bladder' in to_draw and bb['bladder'] is not None):
        # Bladder
        blad = bb['bladder']
        draw.rectangle([blad[1],blad[0],blad[3],blad[2]],outline='green')
    if('rectum' in bb.keys() and 'rectum' in to_draw and bb['rectum'] is not None):
        #Rectum
        rect = bb['rectum']
        draw.rectangle([rect[1],rect[0],rect[3],rect[2]],outline='red')
    if('prostate' in bb.keys() and 'prostate' in to_draw and bb['prostate'] is not None):
        # Prostate
        prost = bb['prostate']
        draw.rectangle([prost[1],prost[0],prost[3],prost[2]],outline='blue')
    if('all' in bb.keys() and 'all' in to_draw and bb['all'] is not None):
        # All
        all = bb['all']
        draw.rectangle([all[1],all[0],all[3],all[2]],outline='yellow')

    if save:
        im.save(im_path.replace('.jpg','-bb.jpg'))
    else:
        im.show()

    del draw

def save_all(path_data,dict,to_draw):
    dir_list = sorted(os.listdir(path_data))
    for filename in dir_list:
        if(filename.endswith('-image.jpg')):
            print('Drawning bounding box for {}'.format(filename))
            draw_bb(os.path.join(path_data,filename),dict[filename]['bb'],to_draw,save=True)


def analyze_results(txt_file,dst_file,total_images,ooi = ['bladder','rectum','prostate']):

    # Using readlines()
    file = open(txt_file, 'r')
    lines = file.readlines()
    # s : score, f : found, r : replicate (when found more than in one in an image and how many more he found)
    scores = {'bladder' : {'s' : 0, 'f' : 0, 'r' : 0},'rectum' : {'s' : 0, 'f' : 0, 'r' : 0}, 'prostate' :{'s' : 0, 'f' : 0, 'r' : 0}, 'all' : {'s' : 0, 'f' : 0, 'r' : 0}}
    count = {'bladder' : 0, 'rectum' : 0, 'prostate' : 0, 'all': 0}
    organs = ['bladder','rectum','prostate','all']
    for line in lines:

        l = line.replace('\n','').split(' ')
        if(l[0] == 'start'):
            count['bladder'] = 0
            count['rectum'] = 0
            count['prostate'] = 0
            count['all'] = 0
        elif(l[0] == 'end'):
            for o in organs:
                if count[o] > 1:
                    scores[o]['r'] += count[o]-1
        else:
            organ = l[0]
            score = float(l[1])
            scores[organ]['s'] += score
            scores[organ]['f'] += 1
            count[organ] += 1

    f = open(dst_file,'w+')
    for o_top in ooi:
        ratio_found = int((scores[o_top]['f']-scores[o_top]['r'])*100/total_images)
        success_score = scores[o_top]['s']/scores[o_top]['f']
        ratio_missed = int((total_images-scores[o_top]['f']+scores[o_top]['r'])*100/total_images)
        overall_score = scores[o_top]['s']/(total_images+scores[o_top]['r'])
        f.write('{} : {}% found, success score of {}, {}% missed, overall score of {}.\n'.format(o_top,ratio_found,success_score,ratio_missed,overall_score))
    f.close()

def learning_graph(history,metrics,legend,scale = 'linear'):
    '''Plots the learning graph.'''

    for m in metrics:
        plt.plot(history[m])

    plt.title('Learning graph')
    plt.ylabel('loss')
    plt.yscale(scale)
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper right')
    plt.show()

def classification_calculator(gt,pred,dim,metric='area'):
    '''Computes TP,TN,FP,FN. dim = (h,w).'''
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
    if metric == 'area':
        TP = interArea
        TN = (totalArea-gtArea-predArea+interArea)
        FP = (predArea-interArea)
        FN = (gtArea-interArea)
    elif metric == 'norm_area':
        TP = interArea/ totalArea
        TN = (totalArea-gtArea-predArea+interArea)/totalArea
        FP = (predArea-interArea)/totalArea
        FN = (gtArea-interArea)/totalArea

    precision = TP/(TP+FP)
    recall = TP/(TP+FN)
    FNR = FN/(TP+FN)

    return (TP,TN,FP,FN,precision,recall,FNR)

def IoU_calculator(box1,box2):
    '''Computes IoU between box1 and box2.'''

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

def average_classification(data_path,pred_file,gt_file,dst_file,ooi,input_dim,output_dim):
    '''Computes the average TP,TN,FP,FN. for all organs.'''
    pred_dict = load_annotation(os.path.join(data_path,pred_file))
    gt_dict= load_annotation(os.path.join(data_path,gt_file))
    ratio = input_dim/output_dim

    # Dictionnaries to keep track of the total TP/TN/FP/FN and the total number of boxes
    classification = {}
    for organ in ooi:
        classification.update({organ : {'TP' : [],'TN' : [],'FP' : [],'FN' : [], 'P' : [], 'R' : [], 'FNR' : []}})

    for filename in gt_dict.keys():
        true_boxes = gt_dict[filename]['bb']
        pred_boxes = pred_dict[filename]

        for o in pred_boxes.keys():
            if len(pred_boxes[o]) > 0:
                if len(pred_boxes[o]) > 1:
                    # Sort from least confident to most confident
                    pred_boxes[o].sort(key=lambda x:x[4])

                pred_box_to_compare = [elem*ratio for elem in pred_boxes[o][-1][0:4]]

                # Computes IoU and updates dict
                (TP,TN,FP,FN,precision,recall,FNR) = classification_calculator(true_boxes[o],pred_box_to_compare,[input_dim,input_dim])
                classification[o]['TP'].append(TP)
                classification[o]['TN'].append(TN)
                classification[o]['FP'].append(FP)
                classification[o]['FN'].append(FN)
                classification[o]['P'].append(precision)
                classification[o]['R'].append(recall)
                classification[o]['FNR'].append(FNR)

    f = open(dst_file,'a')
    for o_toPrint in ooi:
        f.write('({},{},{},{}) average (TP,TN,FP,FN) score for {}.\n'.format(np.mean(classification[o_toPrint]['TP']),np.mean(classification[o_toPrint]['TN']),np.mean(classification[o_toPrint]['FP']),np.mean(classification[o_toPrint]['FN']),o_toPrint))
        f.write('({},{},{},{}) variance (TP,TN,FP,FN) for {}.\n'.format(np.var(classification[o_toPrint]['TP']),np.var(classification[o_toPrint]['TN']),np.var(classification[o_toPrint]['FP']),np.var(classification[o_toPrint]['FN']),o_toPrint))
        f.write('({},{},{}) average (precision,recall,FN rate) score for {}.\n'.format(np.mean(classification[o_toPrint]['P']),np.mean(classification[o_toPrint]['R']),np.mean(classification[o_toPrint]['FNR']),o_toPrint))
        f.write('({},{},{}) variance (precision,recall,FN rate) for {}.\n'.format(np.var(classification[o_toPrint]['P']),np.var(classification[o_toPrint]['R']),np.var(classification[o_toPrint]['FNR']),o_toPrint))
    f.close()

def average_IoU(data_path,pred_file,gt_file,dst_file,dst_iou_file,ooi,input_dim,output_dim):
    '''Computes the average IoU for all organs.'''
    pred_dict = load_annotation(os.path.join(data_path,pred_file))
    gt_dict= load_annotation(os.path.join(data_path,gt_file))
    ratio = input_dim/output_dim # CT/YOLO

    # Dictionnaries to keep track of the total IoU and the total number of boxes
    IoU = {}
    all_iou = {}
    for organ in ooi:
        IoU.update({organ : []})


    for filename in gt_dict.keys():
        true_boxes = gt_dict[filename]['bb']
        pred_boxes = pred_dict[filename]

        # Update filename entry in average IoU
        all_iou.update({filename : {}})

        for o in pred_boxes.keys():
            if len(pred_boxes[o]) > 0:
                if len(pred_boxes[o]) > 1:
                    # Sort from least confident to most confident
                    pred_boxes[o].sort(key=lambda x:x[4])

                pred_box_to_compare = [elem*ratio for elem in pred_boxes[o][-1][0:4]]

                # Computes IoU and updates dict
                iou_val = IoU_calculator(true_boxes[o],pred_box_to_compare)
                IoU[o].append(iou_val)
                all_iou[filename].update({o : iou_val})

    f = open(dst_file,'a')
    for o_toPrint in ooi:
        f.write('{} average IoU score for {} (with variance = {}).\n'.format(np.mean(IoU[o_toPrint]),o_toPrint,np.var(IoU[o_toPrint])))
    f.close()

    save_annotation(all_iou,dst_iou_file)

def case(IoU_path,OOI,dst_file,tolerance = 0.01,bad_thresh = 0.35):
    '''Takes a IoU file with all the slices IoU and identifies an average case'''
    IoU = load_annotation(IoU_path)

    all_iou = {}
    stats = {}
    for organ in OOI :
        all_iou.update({organ : []})
        stats.update({organ : {'mean' : 0, 'var' : 0}})

    for file in IoU.keys():
        for org,val in IoU[file].items():
            all_iou[org].append(val)

    for o in OOI :
        stats[o]['mean'] = np.mean(all_iou[o])
        stats[o]['var'] = np.var(all_iou[o])

    average_cases = []
    bad_cases = []
    missed_cases = []

    for f in IoU.keys():
        count = 0
        for oo in OOI:
            mean = stats[oo]['mean']
            var = stats[oo]['var']
            if(oo not in IoU[f].keys()):
                missed_cases.append(f)
                break
            else:
                if(IoU[f][oo] <= mean+var+tolerance and IoU[f][oo] >= mean-var-tolerance):
                    count += 1
                elif(IoU[f][oo] <= mean-bad_thresh and f not in bad_cases):
                    bad_cases.append(f)

        if(count == len(OOI)):
            average_cases.append(f)

    f = open(dst_file,'w+')
    f.write('AVERAGE : {}\n\n'.format(average_cases))
    f.write('BAD : {}\n\n'.format(bad_cases))
    f.write('MISSED : {}\n\n'.format(missed_cases))
    f.close()

def draw_on_top(data_path,pred_file):
    '''Draws on top of the predicted boxes to see if the annotation file contains correct information.'''
    pred_dict = load_annotation(os.path.join(data_path,pred_file))
    dir_list = sorted(os.listdir(data_path))
    for filename in dir_list:
        if(filename.endswith('.png')):
            index = filename.replace('-pred.png','.jpg')
            for organ in pred_dict[index].keys():
                pred_dict[index][organ].sort(key=lambda x:x[4])
                box = pred_dict[index][organ][-1][0:4]
                draw_single(os.path.join(data_path,filename),box)

def concatenate_history(path):
    '''Concatenates all the history files from path into one dictionnary.'''
    hist_files = os.listdir(path)
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

def concatenate_IoU(path):
    '''Concatenates all the IoU files from path into one dictionnary.'''
    iou_files = os.listdir(path)
    final_iou = {}
    for f in range(len(iou_files)):
        if(os.path.exists(os.path.join(path,'iou'+str(f)+'.p'))):
            h = os.path.join(path,'iou'+str(n)+'.p')
            hx = load_annotation(h)
            if(n == 0):
                final_iou = hx
            else:
                for key,value in hx.items():
                    final_iou[key].extend(value)

    return final_iou

def plot_iou(av_iou_dict,x_axis):
    '''Plots the IoU along with the different epochs.'''

    legend = av_iou_dict.keys()
    for o in legend:
        plt.plot(x_axis,av_iou_dict[o])


    plt.title('Average IoU')
    plt.ylabel('IoU')
    plt.xlabel('epoch')
    plt.legend(legend, loc='upper right')
    plt.show()


if __name__ == '__main__':
    _main()
