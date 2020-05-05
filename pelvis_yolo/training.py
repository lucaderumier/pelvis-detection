'''
Script to train YOLOv2 on pelvic images for bladder, rectum and prostate recognition.

Adapted from github.com/allanzelener/YAD2K by Luca Derumier.
Version 1.0 - May 2020.
'''
import argparse
import os
import pickle
import numpy as np
import PIL
import tensorflow as tf

from keras import optimizers
from keras.layers import Input, Lambda, Conv2D
from keras.models import load_model, Model
from keras.callbacks import TensorBoard, ModelCheckpoint, EarlyStopping
from yad2k.models.keras_yolo import preprocess_true_boxes, yolo_body, yolo_loss
from yad2k.models.keras_yolo import metric
from utils import save_annotation, load_annotation
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
    description="Trains a YOLOv2 model on pelvis custom data.")


argparser.add_argument(
    '-d',
    '--data_path',
    help="path to the folder containing training and validation data folders.",
    default=os.path.join('pelvis_scan','data'))

argparser.add_argument(
    '-tf',
    '--training_file',
    help="name of numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images' for training data (has to be inside the 'data_path/train' folder).",
    default='pelvis_data_train.npz')

argparser.add_argument(
    '-vf',
    '--validation_file',
    help="name of numpy data file (.npz) containing np.object array 'boxes' and np.uint8 array 'images' for training data (has to be inside the 'data_path/val' folder).",
    default='pelvis_data_val.npz')

argparser.add_argument(
    '-r',
    '--results_dir',
    help="path to the folder where the results are going to be stored.",
    default='results')

argparser.add_argument(
    '-m',
    '--model_dir',
    help="path to the folder where the model files are stored.",
    default='model_data')

argparser.add_argument(
    '-w',
    '--weights',
    help="name of the weights file that we want to load (should be in 'models' directory).",
    default='')

argparser.add_argument(
    '-c',
    '--classes',
    help='name of classes txt file (should be in model_dir).',
    default='pelvis_classes.txt')

argparser.add_argument(
    '-G',
    '--generator',
    help="enables custom data generator instead of keras'.",
    action='store_true')

########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    data_path = args.data_path
    training_file = args.training_file
    validation_file = args.validation_file
    results_dir = args.results_dir
    model_dir = args.model_dir
    classes = args.classes
    weights = args.weights
    gen = args.generator

    # Computed arguments
    data_path_train = os.path.join(data_path,'train',training_file)
    data_path_val = os.path.join(data_path,'val',validation_file)
    classes_path = os.path.join(model_dir, classes)

    if(weights == ''):
        weights_path = None
    else:
        weights_path = os.path.join('models',weights)

    # Creating missing directories
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    # Creating config instance
    config = Config()

    # Extracting classes and anchors
    class_names = get_classes(classes_path)
    anchors = config.YOLO_ANCHORS

    # Loading dictionnary
    data_train = np.load(data_path_train,allow_pickle=True)
    data_val = np.load(data_path_val,allow_pickle=True)

    # Extracting images and boxes
    image_data_train, boxes_train = process_data(data_train['images'], data_train['boxes'])
    image_data_val, boxes_val = process_data(data_val['images'], data_val['boxes'])

    # Normalizing data
    normalized_data_train  = normalize(image_data_train,os.path.join(data_path,'train'))
    normalized_data_val = normalize(image_data_val,os.path.join(data_path,'train'),train=False)

    # Creating yolo model
    model_body, model = create_model(anchors, class_names,freeze_body=config.FREEZE,load_pretrained=config.LOAD_PRETRAINED)
    #print(model_body.summary())

    if not gen:

        # Extracting anchor boxes and masks
        detectors_mask_train, matching_true_boxes_train = get_detector_mask(boxes_train, anchors)
        detectors_mask_val, matching_true_boxes_val = get_detector_mask(boxes_val, anchors)

        # Creating input list for validation data
        data_train = [normalized_data_train, boxes_train, detectors_mask_train, matching_true_boxes_train]
        data_val = [normalized_data_val,boxes_val,detectors_mask_val,matching_true_boxes_val]

        # Call to train function
        train(
            model,
            class_names,
            anchors,
            data_train,
            data_val,
            config,
            weights_path=weights_path,
            results_dir=results_dir
        )
    else:
        # Import generators
        from data_generator import DataGenerator

        # Parameters
        params = {'dim': (416,416,3),
              'batch_size': config.BATCH_SIZE,
              'n_classes': 3,
              'n_channels': 1,
              'output_dim' : 1,
              'input_len': 4,
              'shuffle': True}

        # Ids
        list_ids_train = sorted([x for x in os.listdir(os.path.join(data_path,'train')) if x.endswith('.jpg')])
        list_ids_val = sorted([x for x in os.listdir(os.path.join(data_path,'val')) if x.endswith('.jpg')])

        # Data generators
        datagen_train = DataGenerator(list_ids_train, normalized_data_train, anchors, **params)
        datagen_val = DataGenerator(list_ids_val, normalized_data_val, anchors, **params)

        train_generator(
            model,
            datagen_train,
            datagen_val,
            len(list_ids_train),
            image_data_train.shape,
            class_names,
            anchors,
            config,
            weights_path=weights_path,
            results_dir=results_dir
        )


#########################################################
################### Utility functions ###################
#########################################################

def normalize(image_data,training_dir,train=True):
    '''Normalizes the data set by removing specified mean and diving by specified standard deviation.

    Inputs:
        image_data : np array of shape (#images,side,side,channels) containing the images.
        training_dir : the training set directory's path.
        train : whether we are normalizing the training set or another set.

    Returns:
        image_data : the normalized data as a np array of the same shape.
        '''

    n = image_data.shape[0]

    if(train):
        # Normalizaing training set
        mean = np.mean(image_data)
        std = np.std(image_data)
        stat = {'mean' : mean, 'std' : std}
        save_annotation(stat,os.path.join(training_dir,'mean.p'))
    elif(os.path.exists(os.path.join(training_dir,'mean.p'))):
        # Normalizing another set with respect to the training set
        stat = load_annotation(os.path.join(training_dir,'mean.p'))
        mean = stat['mean']
        std = stat['std']
    else:
        raise ValueError('Missing file for training mean and standard deviation.')

    for i in range(n):
        image_data[i] = (image_data[i]-mean)/std

    return image_data


def get_classes(classes_path):
    '''loads the classes.'''
    with open(classes_path) as f:
        class_names = f.readlines()
    class_names = [c.strip() for c in class_names]
    return class_names

def get_anchors(anchors_path):
    '''loads the anchors from a file.'''
    if os.path.isfile(anchors_path):
        with open(anchors_path) as f:
            anchors = f.readline()
            anchors = [float(x) for x in anchors.split(',')]
            return np.array(anchors).reshape(-1, 2)
    else:
        Warning("Could not open anchors file, using default.")
        return YOLO_ANCHORS

def process_data(images, boxes=None):
    '''processes the data.'''
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

#########################################################
#################### Model functions ####################
#########################################################

def create_model(anchors, class_names, freeze_body=True, load_pretrained=True):
    '''
    returns the body of the model and the model

    Inputs:
        anchors: a np array containing the anchor boxes dimension.
        class_names: a list containing the class names.
        load_pretrained: whether or not to load the pretrained model or initialize all weights.
        freeze_body: whether or not to freeze all weights except for the last layer's.

    Returns:
        model_body: YOLOv2 with new output layer.
        model: YOLOv2 with custom loss Lambda layer.
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


    # TODO: Replace Lambda with custom Keras layer for loss.
    model_loss = Lambda(
            yolo_loss,
            output_shape=(1, ),
            name='yolo_loss',
            arguments={'anchors': anchors,
                       'num_classes': len(class_names)})([
                           model_body.output, boxes_input,
                           detectors_mask_input, matching_boxes_input])

    model = Model(
        [model_body.input, boxes_input, detectors_mask_input,
         matching_boxes_input], model_loss)

    return model_body, model

##########################################################
################### Training functions ###################
##########################################################

def train(model, class_names, anchors, data_train, data_val,config,weights_path=None,results_dir='results'):
    ''' Retrains/fine-tune the model.
        Saves the weights every 10 epochs for 200 epochs.

    Inputs:
        model: the model as returned by the create_model function.
        class_names: the class names as loaded by the get_classes function.
        anchors: a np array containing the anchor boxes dimensions.
        data_train: a list containing all the training input data for the network.
        data_val: a list containing all the validation input data for the network.
        config: a config object that has all the configuration parameters for the training.
        weights_path: the path to the weights that the models needs to load. None if we want to start from scratch.
        results_dir: the path to the directory where the results are going to be saved.
    '''

    # Creating missing directories
    if not os.path.exists(os.path.join(results_dir,'models')):
        os.makedirs(os.path.join(results_dir,'models'))
    if not os.path.exists(os.path.join(results_dir,'history')):
        os.makedirs(os.path.join(results_dir,'history'))

    # Loading configuration parameters
    load_pretrained = config.LOAD_PRETRAINED
    freeze_body = config.FREEZE
    learning_rate = config.LEARNING_RATE
    batch_size = config.BATCH_SIZE
    epochs = config.EPOCHS

    ######################################
    ############# TRAINING ###############
    ######################################

    # Defining metrics
    met = metric([model.get_layer('conv2d_24').output,model.inputs[1],model.inputs[2], model.inputs[3]],data_train[0].shape,anchors,len(class_names),score_threshold=0.07,iou_threshold=0.0)

    # Compile model and load weights_name
    optimizer = optimizers.Adam(lr = learning_rate)

    model.compile(
        optimizer=optimizer, loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        },metrics=[met.IoU,met.classification_loss,met.coord_loss,met.conf_loss])  # This is a hack to use the custom loss function in the last layer

    if weights_path is not None:
        model.load_weights(weights_path)

    # Callbacks
    logging = tf.keras.callbacks.TensorBoard()
    #checkpoint = ModelCheckpoint("trained_best.h5", monitor='val_loss',save_weights_only=True, save_best_only=True)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    # Training for loop
    for stage in range(20):
        # SLaunch training
        history = model.fit(data_train,
                      np.zeros(len(data_train[0])),
                      validation_data=(data_val,np.zeros(len(data_val[0]))),
                      batch_size=batch_size,
                      epochs=epochs,
                      callbacks=[logging])

        # Saving history and weights
        with open(os.path.join(results_dir,'history','history'+str(stage)+'.p'), 'wb') as fp:
            pickle.dump(history.history, fp)
        model.save_weights(os.path.join(results_dir,'models','trained_stage_'+str(stage)+'.h5'))

        # Safety load
        model.load_weights(os.path.join(results_dir,'models','trained_stage_'+str(stage)+'.h5'))

def train_generator(model, training_generator, validation_generator, train_size, data_shape,class_names, anchors,config,weights_path = None,results_dir='results'):
    ''' Retrains/fine-tune the model with custom data generator.
        Saves the weights every 10 epochs for 200 epochs.

    Inputs:
        model: the model as returned by the create_model function.
        training_generator: the generator object for training data.
        validation_generator: the generator object for validation data.
        train_size: the size of the training set.
        data_shape: the shape of the input data.
        class_names: the list contatining the class names.
        anchors: a np array containing the anchor boxes dimensions.
        config: a config object that has all the configuration parameters for the training.
        weights_path: the path to the weights that the models needs to load. None if we want to start from scratch.
        results_dir: the path to the directory where the results are going to be saved.
    '''

    # Creating missing directories
    if not os.path.exists(os.path.join(results_dir,'models')):
        os.makedirs(os.path.join(results_dir,'models'))
    if not os.path.exists(os.path.join(results_dir,'history')):
        os.makedirs(os.path.join(results_dir,'history'))

    # Loading configuration parameters
    freeze_body = config.FREEZE
    learning_rate = config.LEARNING_RATE
    batch_size = config.BATCH_SIZE
    epochs = config.EPOCHS

    ######################################
    ############# TRAINING ###############
    ######################################

    # Defining metrics
    met = metric([model.get_layer('conv2d_24').output,model.inputs[1],model.inputs[2], model.inputs[3]],data_shape,anchors,len(class_names),score_threshold=0.07,iou_threshold=0.0)

    # Compile model and load weights_name
    optimizer = optimizers.Adam(lr = learning_rate)

    model.compile(
        optimizer=optimizer, loss={
            'yolo_loss': lambda y_true, y_pred: y_pred
        },metrics=[met.IoU,met.classification_loss,met.coord_loss,met.conf_loss])  # This is a hack to use the custom loss function in the last layer

    if weights_path is not None:
        model.load_weights(weights_path)

    # Callbacks
    logging = tf.keras.callbacks.TensorBoard()
    #checkpoint = ModelCheckpoint("trained_best.h5", monitor='val_loss',save_weights_only=True, save_best_only=True)
    #early_stopping = EarlyStopping(monitor='val_loss', min_delta=0, patience=15, verbose=1, mode='auto')

    # Training for loop
    for stage in range(20):
        # SLaunch training
        history = model.fit_generator(generator=training_generator,
                        validation_data=validation_generator,
                        steps_per_epoch=int(np.ceil(train_size/batch_size)),
                        epochs=epochs,
                        callbacks=[logging])

        # Saving history and weights
        with open(os.path.join(results_dir,'history','history'+str(stage)+'.p'), 'wb') as fp:
            pickle.dump(history.history, fp)
        model.save_weights(os.path.join(results_dir,'models','trained_stage_'+str(stage)+'.h5'))

        # Safety load
        model.load_weights(os.path.join(results_dir,'models','trained_stage_'+str(stage)+'.h5'))

############################################
################### Main ###################
############################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
