'''
Script to train YOLO on pelvic images for bladder, rectum and prostate recognition.
'''
import argparse
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import PIL
import tensorflow as tf

from keras import backend as K
from keras import optimizers
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from yad2k.models.keras_yolo import (preprocess_true_boxes, yolo_body, yolo_eval, yolo_head, yolo_loss)
from yad2k.models.keras_yolo import metric
from yad2k.utils.draw_boxes import draw_boxes

# Sets the environment to use GPU = 3 only
gpu = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

# Mode of exectuion
# DONT FORTGET TO RUN python3 pelvis_training.py | tee RESULTS/results.txt
MODE = 'extract' #'train', 'test', 'try', 'test_on_train', 'test_on_val' or 'extract'

# Data directories
if(MODE == 'try'):
    DATASET_DIR = os.path.join('pelvis_scan','data_TEST')
elif(MODE == 'extract'):
    DATASET_DIR = os.path.join('pelvis_scan','EXTRACT_TEST')
    #DATASET_DIR = os.path.join('pelvis_scan','FULL_IMAGES_CT')
else:
    DATASET_DIR = os.path.join('pelvis_scan','DATA_CT')


MODEL_DATA_DIR = 'model_data'
RESULTS_DIR = 'RESULTS'

# Files for annotations
DATASET_FILE_TRAIN = 'pelvis_data_train.npz'
DATASET_FILE_VAL = 'pelvis_data_val.npz'
DATASET_FILE_TEST = 'pelvis_data_test.npz'

# Previously trained weights path
WEIGHTS = os.path.join(RESULTS_DIR,'models','trained_stage_20.h5') # CHANGES THIS ONE FOR DRAWING RESULTS ON DIFFERENT EPOCH

# Variables for configurations
class Config():
    '''Configuration for training.'''
    FREEZE = False # Freeze all layers but last
    LOAD_PRETRAINED = False # Loads pre-trained yolo.h5
    LOAD_PREVIOUS_MODEL = True # Loads a previously trained model (WIEGHTS)
    NON_BEST_SUP = True
    LEARNING_RATE = 0.001
    YOLO_ANCHORS = np.array(
        ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
         (7.88282, 3.52778), (9.77052, 9.16828)))

    '''
    #K-MEAN-CLUSTERING ANCHORS
    YOLO_ANCHORS = np.array(
        ((6.13000678, 4.92235932), (2.3183523, 2.56730623), (1.43822055, 1.44648445),
         (4.31028148, 3.67089778), (2.39533622, 5.34201443)))
    '''


def _main():
    # Creating config instance
    config = Config()

    # Creating missing directories
    if not os.path.exists(RESULTS_DIR):
        os.makedirs(RESULTS_DIR)

    # Defining full data paths
    if(MODE != 'extract'):
        data_path_train = os.path.join(DATASET_DIR,'train', DATASET_FILE_TRAIN)
        data_path_val = os.path.join(DATASET_DIR,'val', DATASET_FILE_VAL)
        data_path_test = os.path.join(DATASET_DIR,'test', DATASET_FILE_TEST)
    classes_path = os.path.join(MODEL_DATA_DIR, 'pelvis_classes.txt')
    weights_path = WEIGHTS

    # Extracting classes and anchors
    class_names = get_classes(classes_path)
    anchors = config.YOLO_ANCHORS

    # Loads custom (pelvis) data
    if(MODE == 'extract'):
        # Creating model and printing summary
        model_body, model = create_model(anchors, class_names,freeze_body=config.FREEZE,load_pretrained=config.LOAD_PRETRAINED)

        for d in os.listdir(DATASET_DIR):
            if(d.startswith('charleroi') or d.startswith('namur')):
                print('NEW_DIRECTORY_ITERAION')
                # iterating through image folders
                data_test = np.load(os.path.join(DATASET_DIR,d,'pelvis_data.npz'),allow_pickle=True)
                # Extracting images and boxes
                image_data_test, boxes_test = process_data(data_test['images'], data_test['boxes'])
                # Extracting anchor boxes and masks
                detectors_mask_test, matching_true_boxes_test = get_detector_mask(boxes_test, anchors)
                # Defining result images path
                out_path = os.path.join(RESULTS_DIR,'full_images',d)
                draw(model_body,
                    class_names,
                    anchors,
                    image_data_test,
                    image_set='all',
                    weights_name=weights_path,
                    non_best_sup=config.NON_BEST_SUP,
                    save_all=True,
                    out_path=out_path)

    elif(MODE == 'test' or MODE == 'test_on_train' or MODE == 'test_on_val'):
        # Loading dictionnary
        if(MODE == 'test' or MODE == 'extract'):
            data_test = np.load(data_path_test,allow_pickle=True)
        elif(MODE == 'test_on_train'):
            data_test = np.load(data_path_train,allow_pickle=True)
        elif(MODE == 'test_on_val'):
            data_test = np.load(data_path_val,allow_pickle=True)
        else:
            raise ValueError('Unknown mode : {}. Should be test, test_on_train, test_on_val or extract.'.format(MODE))

        # Extracting images and boxes
        image_data_test, boxes_test = process_data(data_test['images'], data_test['boxes'])

        # Extracting anchor boxes and masks
        detectors_mask_test, matching_true_boxes_test = get_detector_mask(boxes_test, anchors)

        # Defining result images path
        out_path = os.path.join(RESULTS_DIR,'images')

        # Creating model and printing summary
        model_body, model = create_model(anchors, class_names,freeze_body=config.FREEZE,load_pretrained=config.LOAD_PRETRAINED)
        draw(model_body,
            class_names,
            anchors,
            image_data_test,
            image_set='all',
            weights_name=weights_path,
            non_best_sup=config.NON_BEST_SUP,
            save_all=True,
            out_path=out_path)

    else:
        # Loading dictionnary
        data_train = np.load(data_path_train,allow_pickle=True)
        data_val = np.load(data_path_val,allow_pickle=True)

        # Extracting images and boxes
        image_data_train, boxes_train = process_data(data_train['images'], data_train['boxes'])
        image_data_val, boxes_val = process_data(data_val['images'], data_val['boxes'])

        # Extracting anchor boxes and masks
        detectors_mask_train, matching_true_boxes_train = get_detector_mask(boxes_train, anchors)
        detectors_mask_val, matching_true_boxes_val = get_detector_mask(boxes_val, anchors)

        model_body, model = create_model(anchors, class_names,freeze_body=config.FREEZE,load_pretrained=config.LOAD_PRETRAINED)
        #print(model_body.summary())

        if(MODE == 'try'):
            data_val = None
        elif(MODE == 'train'):
            data_val = [image_data_val,boxes_val,detectors_mask_val,matching_true_boxes_val]

        train(
            model,
            class_names,
            anchors,
            image_data_train,
            boxes_train,
            detectors_mask_train,
            matching_true_boxes_train,
            data_val,
            config,
            weights_path=weights_path,
            results_dir=RESULTS_DIR
        )


def get_classes(classes_path):
    '''loads the classes'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def process_data(images, boxes=None):
    '''processes the data'''
    images = [PIL.Image.fromarray(i) for i in images]
    orig_size = np.array([images[0].width, images[0].height])
    orig_size = np.expand_dims(orig_size, axis=0)

    # Image preprocessing.
    processed_images = [i.resize((416, 416), PIL.Image.BICUBIC) for i in images]
    processed_images = [np.array(image, dtype=np.float) for image in processed_images]
    processed_images = [image/255. for image in processed_images]

    if boxes is not None:
        # Box preprocessing.
        # Original boxes stored as 1D list of class, x_min, y_min, x_max, y_max.
        boxes = [box.reshape((-1, 5)) for box in boxes]
        # Get extents as y_min, x_min, y_max, x_max, class for comparision with
        # model output.
        boxes_extents = [box[:, [2, 1, 4, 3, 0]] for box in boxes]

        # Get box parameters as x_center, y_center, box_width, box_height, class.
        boxes_xy = [0.5 * (box[:, 3:5] + box[:, 1:3]) for box in boxes]
        boxes_wh = [box[:, 3:5] - box[:, 1:3] for box in boxes]
        boxes_xy = [boxxy / orig_size for boxxy in boxes_xy]
        boxes_wh = [boxwh / orig_size for boxwh in boxes_wh]
        boxes = [np.concatenate((boxes_xy[i], boxes_wh[i], box[:, 0:1]), axis=1) for i, box in enumerate(boxes)]

        # find the max number of boxes
        max_boxes = 0
        for boxz in boxes:
            if boxz.shape[0] > max_boxes:
                max_boxes = boxz.shape[0]

        # add zero pad for training
        for i, boxz in enumerate(boxes):
            if boxz.shape[0]  < max_boxes:
                zero_padding = np.zeros( (max_boxes-boxz.shape[0], 5), dtype=np.float32)
                boxes[i] = np.vstack((boxz, zero_padding))

        return np.array(processed_images), np.array(boxes)
    else:
        return np.array(processed_images)

def get_detector_mask(boxes, anchors):
    '''
    Precompute detectors_mask and matching_true_boxes for training.
    Detectors mask is 1 for each spatial position in the final conv layer and
    anchor that should be active for the given boxes and 0 otherwise.
    Matching true boxes gives the regression targets for the ground truth box
    that caused a detector to be active or 0 otherwise.
    '''
    detectors_mask = [0 for i in range(len(boxes))]
    matching_true_boxes = [0 for i in range(len(boxes))]
    for i, box in enumerate(boxes):
        detectors_mask[i], matching_true_boxes[i] = preprocess_true_boxes(box, anchors, [416, 416])

    return np.array(detectors_mask), np.array(matching_true_boxes)

def create_model(anchors, class_names, freeze_body=True, load_pretrained=True):
    '''
    returns the body of the model and the model

    # Params:

    load_pretrained: whether or not to load the pretrained model or initialize all weights

    freeze_body: whether or not to freeze all weights except for the last layer's

    # Returns:

    model_body: YOLOv2 with new output layer

    model: YOLOv2 with custom loss Lambda layer

    '''

    detectors_mask_shape = (13, 13, 5, 1)
    matching_boxes_shape = (13, 13, 5, 5)

    # Create model input layers.
    image_input = Input(shape=(416, 416, 3))
    boxes_input = Input(shape=(None, 5))
    detectors_mask_input = Input(shape=detectors_mask_shape)
    matching_boxes_input = Input(shape=matching_boxes_shape)

    # Create model body.
    yolo_model = yolo_body(image_input, len(anchors), len(class_names))
    topless_yolo = Model(yolo_model.input, yolo_model.layers[-2].output)

    if load_pretrained:
        # Save topless yolo:
        topless_yolo_path = os.path.join('model_data', 'yolo_topless.h5')
        if not os.path.exists(topless_yolo_path):
            print("CREATING TOPLESS WEIGHTS FILE")
            yolo_path = os.path.join('model_data', 'yolo.h5')
            model_body = load_model(yolo_path)
            model_body = Model(model_body.inputs, model_body.layers[-2].output)
            model_body.save_weights(topless_yolo_path)
        topless_yolo.load_weights(topless_yolo_path)


    if freeze_body:
        for layer in topless_yolo.layers:
            layer.trainable = False
    final_layer = Conv2D(len(anchors)*(5+len(class_names)), (1, 1), activation='linear')(topless_yolo.output)

    model_body = Model(image_input, final_layer)

    # Place model loss on CPU to reduce GPU memory usage.
    with tf.device('/cpu:0'):
        # TODO: Replace Lambda with custom Keras layer for loss.
        model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input
                       ])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model



def train(model, class_names, anchors, image_data, boxes, detectors_mask, matching_true_boxes,data_val,config,validation_split=0.1,weights_path = '',results_dir=''):
    '''
    retrain/fine-tune the model

    logs training with tensorboard

    saves training weights in current directory

    best weights according to val_loss is saved as trained_stage_3_best.h5
    '''

    if not os.path.exists(os.path.join(results_dir,'models')):
        os.makedirs(os.path.join(results_dir,'models'))
    if not os.path.exists(os.path.join(results_dir,'history')):
        os.makedirs(os.path.join(results_dir,'history'))

    load_previous_model=config.LOAD_PREVIOUS_MODEL
    load_pretrained=config.LOAD_PRETRAINED
    freeze_body=config.FREEZE
    learning_rate = config.LEARNING_RATE

    ###################################
    ############# EPOCH 0 #############
    ###################################

    stage = '0'

    met = metric([model.get_layer('conv2d_24').output,model.inputs[1],model.inputs[2], model.inputs[3]],image_data.shape,anchors,len(class_names),score_threshold=0.07,iou_threshold=0.0)

    optimizer = optimizers.Adam(learning_rate = learning_rate)
    model.compile(
        optimizer=optimizer, loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        },metrics=[met.IoU,met.classification_loss,met.coord_loss,met.conf_loss])  # This is a hack to use the custom loss function in the last layer

    if load_previous_model:
        model.load_weights(weights_path)

    logging = TensorBoard()
    checkpoint = ModelCheckpoint("trained_best.h5", monitor='val_loss',
                                 save_weights_only=True, save_best_only=True)

    early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')


    # Split the data
    if data_val is None:
        history0 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])
    else:
        history0 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_data=(data_val,np.zeros(len(data_val[0]))),
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])

    # Saving history
    with open(os.path.join(results_dir,'history','history0.p'), 'wb') as fp:
        pickle.dump(history0.history, fp)

    model.save_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    ###################################
    ############# EPOCH 5 #############
    ###################################

    #model_body, model = create_model(anchors, class_names, load_pretrained=load_pretrained, freeze_body=freeze_body)
    #met2 = metric([model.get_layer('conv2d_48').output,model.inputs[1],model.inputs[2], model.inputs[3]],image_data.shape,anchors,len(class_names),score_threshold=0.07,iou_threshold=0.0)

    model.load_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    stage = '1'

    if data_val is None:
        history1 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])
    else:
        history1 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_data=(data_val,np.zeros(len(data_val[0]))),
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])

    # Saving history
    with open(os.path.join(results_dir,'history','history1.p'), 'wb') as fp:
        pickle.dump(history1.history, fp)


    model.save_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    ###################################
    ############# EPOCH 10 #############
    ###################################

    model.load_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    stage = '2'

    if data_val is None:
        history2 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])
    else:
        history2 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_data=(data_val,np.zeros(len(data_val[0]))),
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])

    # Saving history
    with open(os.path.join(results_dir,'history','history2.p'), 'wb') as fp:
        pickle.dump(history2.history, fp)


    model.save_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    ####################################
    ############# EPOCH 15 #############
    ####################################

    model.load_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    stage = '3'

    if data_val is None:
        history3 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])
    else:
        history3 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_data=(data_val,np.zeros(len(data_val[0]))),
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])

    # Saving history
    with open(os.path.join(results_dir,'history','history3.p'), 'wb') as fp:
        pickle.dump(history3.history, fp)


    model.save_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    ####################################
    ############# EPOCH 20 #############
    ####################################

    model.load_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    stage = '4'

    if data_val is None:
        history4 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])
    else:
        history4 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_data=(data_val,np.zeros(len(data_val[0]))),
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])

    # Saving history
    with open(os.path.join(results_dir,'history','history4.p'), 'wb') as fp:
        pickle.dump(history4.history, fp)


    model.save_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    ####################################
    ############# EPOCH 25 #############
    ####################################

    model.load_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    stage = '5'

    if data_val is None:
        history5 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])
    else:
        history5 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_data=(data_val,np.zeros(len(data_val[0]))),
                  batch_size=32,
                  epochs=5,
                  callbacks=[logging])

    # Saving history
    with open(os.path.join(results_dir,'history','history5.p'), 'wb') as fp:
        pickle.dump(history5.history, fp)

    model.save_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    ####################################
    ############# EPOCH 30 #############
    ####################################

    model.load_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    stage = '6'

    if data_val is None:
        history6 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=32,
                  epochs=30,
                  callbacks=[logging])
    else:
        history6 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_data=(data_val,np.zeros(len(data_val[0]))),
                  batch_size=32,
                  epochs=30,
                  callbacks=[logging])

    # Saving history
    with open(os.path.join(results_dir,'history','history6.p'), 'wb') as fp:
        pickle.dump(history6.history, fp)

    model.save_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    ####################################
    ############# EPOCH 60 #############
    ####################################

    model.load_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    stage = '7'

    if data_val is None:
        history7 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=32,
                  epochs=30,
                  callbacks=[logging])
    else:
        history7 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_data=(data_val,np.zeros(len(data_val[0]))),
                  batch_size=32,
                  epochs=30,
                  callbacks=[logging, checkpoint, early_stopping])

    # Saving history
    with open(os.path.join(results_dir,'history','history7.p'), 'wb') as fp:
        pickle.dump(history7.history, fp)

    model.save_weights(os.path.join(results_dir,'models','trained_stage_'+stage+'.h5'))

    '''

    if data_val is None:
        history3 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_split=validation_split,
                  batch_size=8,
                  epochs=30,
                  callbacks=[logging, checkpoint, early_stopping])
    else:
        history3 = model.fit([image_data, boxes, detectors_mask, matching_true_boxes],
                  np.zeros(len(image_data)),
                  validation_data=(data_val,np.zeros(len(data_val[0]))),
                  batch_size=8,
                  epochs=30,
                  callbacks=[logging, checkpoint, early_stopping])


    model.save_weights(os.path.join(results_dir,'models','trained_stage_3.h5'))

    # Saving history
    with open(os.path.join(results_dir,'history','history3.p'), 'wb') as fp:
        pickle.dump(history3.history, fp)
    '''


def draw(model_body, class_names, anchors, image_data, image_set='val',
            weights_name='trained_stage_3_best.h5',non_best_sup=False, out_path="output_images", save_all=True):
    '''
    Draw bounding boxes on image data
    '''

    if image_set == 'train':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[:int(len(image_data)*.9)]])
    elif image_set == 'val':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data[int(len(image_data)*.9):]])
    elif image_set == 'all':
        image_data = np.array([np.expand_dims(image, axis=0)
            for image in image_data])
    else:
        ValueError("draw argument image_set must be 'train', 'val', or 'all'")

    model_body.load_weights(weights_name)

    # Create output variables for prediction.
    yolo_outputs = yolo_head(model_body.output, anchors, len(class_names))
    input_image_shape = K.placeholder(shape=(2, ))
    boxes, scores, classes = yolo_eval(
        yolo_outputs, input_image_shape, score_threshold=0.07, iou_threshold=0.0)

    # Dictionnary to export the predicted bounding boxes
    boxes_dict = {}

    # Run prediction on overfit image.
    sess = K.get_session()  # TODO: Remove dependence on Tensorflow session.

    if  not os.path.exists(out_path):
        os.makedirs(out_path)
    for i in range(len(image_data)):
        print('start')
        out_boxes, out_scores, out_classes = sess.run(
            [boxes, scores, classes],
            feed_dict={
                model_body.input: image_data[i],
                input_image_shape: [image_data.shape[2], image_data.shape[3]],
                K.learning_phase(): 0
            })

        if non_best_sup:
            assert len(out_classes) == len(out_scores)
            assert len(out_boxes) == len(out_scores)
            new_out_classes = []
            new_out_boxes = []
            new_out_scores = []
            for idx in range(len(out_classes)):
                if(out_classes[idx] not in new_out_classes):
                    new_out_classes.append(out_classes[idx])
                    new_out_boxes.append(out_boxes[idx])
                    new_out_scores.append(out_scores[idx])
                elif(new_out_scores[new_out_classes.index(out_classes[idx])] < out_scores[idx]):
                    swipe_idx = new_out_classes.index(out_classes[idx])
                    new_out_scores[swipe_idx] = out_scores[idx]
                    new_out_boxes[swipe_idx] = out_boxes[idx]

            new_out_boxes = np.asarray(new_out_boxes)
            new_out_scores = np.asarray(new_out_scores)
            new_out_classes = np.asarray(new_out_classes)

            # Plot image with predicted boxes.
            image_with_boxes = draw_boxes(image_data[i][0], new_out_boxes, new_out_classes,
                                        class_names, new_out_scores)
        else:
            # Plot image with predicted boxes.
            image_with_boxes = draw_boxes(image_data[i][0], out_boxes, out_classes,
                                        class_names, out_scores)


        # Updates dictionnary
        boxes_dict.update({i : {}})
        for c in class_names:
            boxes_dict[i].update({c : []})
        for j in range(len(out_boxes)):
            organ = class_names[out_classes[j]]
            new_box = list(out_boxes[j])
            new_box.append(out_scores[j])
            boxes_dict[i][organ].append(new_box)

        print('end')


        # Save the image:
        if save_all or (len(out_boxes) > 0):
            image = PIL.Image.fromarray(image_with_boxes)
            image.save(os.path.join(out_path,str(i)+'.png'))



        # To display (pauses the program):
        # plt.imshow(image_with_boxes, interpolation='nearest')
        # plt.show()

    # Saving boxes
    with open(os.path.join(out_path,'pred_boxes.p'), 'wb') as fp:
        pickle.dump(boxes_dict, fp)



if __name__ == '__main__':
    _main()
