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

from keras import backend as K
from yad2k.models.keras_yolo import yolo_eval, yolo_head
from yad2k.utils.draw_boxes import draw_boxes
from utils import save_annotation, load_annotation
from training import normalize, get_classes, process_data, get_detector_mask, create_model
from config import Config

########################################################
#################### GPU Constraint ####################
########################################################

gpu = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

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



#########################################################
######################## Predict ########################
#########################################################

def predict(model_body, class_names, anchors, image_data, weights_name, dir_list, non_best_sup=False, results_dir='results', save=False):
    '''Runs the detection algorithm on image_data.

    Inputs:
        model_body: the body of the model as returned by the create_model function.
        class_names: a list containing the class names.
        anchors: a np array containing the anchor boxes dimension.
        image_data: np array of shape (#images,side,side,channels) containing the images.
        weights_name: the name of the weight file that we want to load.
        non_best_sup: wether or not to perform non best suppression during predictions.
        results_dir: directory where the results will be saved.
        save: wether or not to save the output images.

    Returns:
        boxes_dict: the dictionnary containing the bounding boxes and scores.
                    boxes_dict = {filename : {'bladder': [[xA,yA,xB,yB,score],[...]],
                                              'rectum': [..],
                                              'prostate': [..]},
                                  filename : {...},...}
    '''

    # Creating missing directories
    if  not os.path.exists(results_dir):
        os.makedirs(results_dir)
    if  not os.path.exists(os.path.join(results_dir,'images')):
        os.makedirs(os.path.join(results_dir,'images'))
    if  not os.path.exists(os.path.join(results_dir,'predictions')):
        os.makedirs(os.path.join(results_dir,'predictions'))

    # Loading image data in the right format
    image_data = np.array([np.expand_dims(image, axis=0) for image in image_data])

    # Loading weights
    model_body.load_weights(os.path.join('models',weights_name))

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0.0)

    # Dictionnary to export the predicted bounding boxes
    boxes_dict = {}

    # Run prediction
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    for i in range(len(image_data)):
        print('predicting boxes for {}'.format(dir_list[i]))
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })

        if non_best_sup:
            (new_out_boxes,new_out_classes,new_out_scores) = non_best_suppression(out_boxes,out_classes,out_scores)

            # Plot image with predicted boxes.
            if save:
                image_with_boxes = draw_boxes(image_data[i][0], new_out_boxes, new_out_classes,
                                        class_names, new_out_scores)
                image = PIL.Image.fromarray(image_with_boxes)
                image.save(os.path.join(results_dir,'images',dir_list[i]+'.png'))
        elif save:
            # Plot image with predicted boxes.
            image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                        class_names, out_scores)
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(results_dir,'images',dir_list[i]+'.png'))

        # Updates dictionnary
        boxes_dict.update({dir_list[i] : {}})
        for c in class_names:
            boxes_dict[dir_list[i]].update({c : []})
        for j in range(len(out_boxes)):
            organ = class_names[out_classes[j]]
            new_box = list(out_boxes[j])
            new_box.append(out_scores[j])
            boxes_dict[dir_list[i]][organ].append(new_box)

    # Saving boxes
    return boxes_dict


############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
