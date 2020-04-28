import os
os.environ["CUDA_VISIBLE_DEVICES"] = "3"

import numpy as np
import tensorflow as tf
import pickle

from pelvis_training import process_data, get_detector_mask

class DataGenerator(tf.keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self,list_IDs, data, anchors, batch_size=32, dim=(416,416,3),input_len=4,output_dim=1, n_channels=1,
             n_classes=3, shuffle=True):
        'Initialization'
        self.dim = dim
        self.list_IDs = list_IDs
        self.input_len = input_len
        self.data = data
        self.anchors = anchors
        self.batch_size = batch_size
        self.output_dim = output_dim
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generates one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        #X, Y = self.__data_generation(list_IDs_temp)
        X, Y = self.__data_generation(indexes)
        return X, Y

    def __data_generation(self,indexes):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)

        image_data, boxes = process_data(self.data['images'], self.data['boxes'])
        detectors_mask, matching_true_boxes = get_detector_mask(boxes, self.anchors)

        # Empty input/output arrays initialization
        X_image_data = np.empty((self.batch_size, image_data.shape[1],image_data.shape[2],image_data.shape[3]))
        X_boxes = np.empty((self.batch_size, boxes.shape[1],boxes.shape[2]))
        X_detectors_mask = np.empty((self.batch_size, detectors_mask.shape[1],detectors_mask.shape[2],detectors_mask.shape[3],detectors_mask.shape[4]))
        X_matching_true_boxes = np.empty((self.batch_size, matching_true_boxes.shape[1], matching_true_boxes.shape[2], matching_true_boxes.shape[3], matching_true_boxes.shape[4]))
        Y = np.empty((self.batch_size, self.output_dim))

        for i,idx in enumerate(indexes):
            X_image_data[i,] = image_data[idx]
            X_boxes[i,] = boxes[idx]
            X_detectors_mask[i,] = detectors_mask[idx]
            X_matching_true_boxes[i,] = matching_true_boxes[idx]
            Y[i,] = 0

        return [X_image_data,X_boxes,X_detectors_mask,X_matching_true_boxes],Y


if __name__ == '__main__':
    val_path = os.path.join('pelvis_scan','DATA_CT_SPARSE','val')
    list_ids = sorted([x for x in os.listdir(val_path) if x.endswith('.jpg')])

    # Loading dictionnary
    data_val = np.load(os.path.join(val_path,'pelvis_data_val.npz'),allow_pickle=True)

    # Anchors
    anchors = np.array(
        ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
         (7.88282, 3.52778), (9.77052, 9.16828)))

    # Parameters
    params = {'dim': (416,416,3),
          'batch_size': 32,
          'n_classes': 3,
          'n_channels': 1,
          'output_dim' : 1,
          'input_len': 4,
          'shuffle': True}

    datagen = DataGenerator(list_ids, data_val, anchors, **params)
    datagen.__getitem__(0)
