'''
Script of base configuration class.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import numpy as np

#########################################################
######################## Configs ########################
#########################################################

class Config():
    '''Base configuration class. For custom configurations, create a
    sub-class that inherits from this one and override properties
    that need to be changed.'''

    # Freeze all layers but last
    FREEZE = False

    # Loads pre-trained weights from training on pascal data set
    LOAD_PRETRAINED = False

    # Non best suppression
    NON_BEST_SUP = False

    # Learning rate
    LEARNING_RATE = 0.0001

    # Batch size
    BATCH_SIZE = 32

    # Anchors
    YOLO_ANCHORS = np.array(
        ((0.57273, 0.677385), (1.87446, 2.06253), (3.33843, 5.47434),
         (7.88282, 3.52778), (9.77052, 9.16828)))

    '''
    #K-MEAN-CLUSTERING ANCHORS
    YOLO_ANCHORS = np.array(
        ((6.13000678, 4.92235932), (2.3183523, 2.56730623), (1.43822055, 1.44648445),
         (4.31028148, 3.67089778), (2.39533622, 5.34201443)))
    '''

    # Input/Output ratio size
    INPUT_DIM = (500,500)
    OUTPUT_DIM = (416,416)
