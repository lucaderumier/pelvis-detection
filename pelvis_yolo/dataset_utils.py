'''
Script for transforming our jpg images dataset into a npz file usable by yolo.

Written by Luca Derumier.
Version 1.0 - May 2020.
'''
import argparse
import numpy as np
import os
import pickle
import PIL.Image
from skimage.color import gray2rgb
from utils import load_annotation

########################################################
#################### GPU Constraint ####################
########################################################

gpu = 3
os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu)

#########################################################
################### Parsing arguments ###################
#########################################################

argparser = argparse.ArgumentParser(
    description="Transforms an image dataset into a npz file usable by yolo.")

argparser.add_argument(
    '-d',
    '--data_path',
    help="path to the folder containing training and validation data folders.",
    default=os.path.join('pelvis_scan','data'))

########################################################
######################### Main #########################
########################################################

def _main(args):
    # Raw arguments from parser
    data_path = args.data_path

    # Dictionnary of the organs and their index in the classes.txt file
    label_dict = {'bladder' : 0, 'rectum' : 1, 'prostate' : 2}

    # Environment variables
    sets = [x for x in ['train','val','test'] if x in os.listdir(data_path)]
    if len(sets) == 0:
        raise ValueError('No set to extract the data from.')

    for set in sets:
        dataset_dir = os.path.join(data_path,set)
        annotation = 'annotations_'+set+'.p'
        filename = 'pelvis_data_'+set+'.npz'
        dict = load_annotation(os.path.join(dataset_dir,annotation))
        yolo_dataset(dict,dataset_dir,filename,label_dict)

#############################################
################### Utils ###################
#############################################

def yolo_dataset(dict,path,npz_filename,label_dict,shuffle = False):
    '''Transform our dictionnary and images to npz data file that yolo can use.

    Inputs:
        dict: dictionnary with the bounding box annotations
        path: the paht to the folder that has the images.
        npz_filename: the name we want for our output file.
        label_dict: dictionnary that contains the organs as keys and their correspong class values as values.
        shuffle: wether to shuffle the dataset or not.'''

    dir_list = sorted(os.listdir(path))
    images = []
    size = len([f for f in dir_list if(f.endswith('.jpg'))])
    image_labels = []

    # Loads images and stores bounding boxes
    for filename in dir_list:
        if(filename.endswith('.jpg')):

            # Loads the image and
            image = PIL.Image.open(os.path.join(path,filename))

            # Convert the image to RGB foramt (needed for yolo)
            rgbimg = PIL.Image.new("RGB", image.size)
            rgbimg.paste(image)
            img = np.array(rgbimg, dtype=np.uint8)


            images.append(img)

            all_boxes = []

            for organ,index in label_dict.items():
                if(organ in dict[filename]['bb'].keys() and dict[filename]['bb'][organ] is not None):
                    print('Extracting {} box for {}'.format(organ,filename))
                    box = dict[filename]['bb'][organ]
                    organ_index = label_dict[organ]
                    xA = box[1]
                    yA = box[0]
                    xB = box[3]
                    yB = box[2]

                    if(xA > 0 or yA > 0 or xB < image.width-1 or yB != image.height-1):
                        new_box = np.array([organ_index,xA,yA,xB,yB])
                        all_boxes.append(new_box)


            print(all_boxes)
            image_labels.append(np.array(all_boxes))

    # Convert list to numpy array for saving
    images = np.array(images, dtype=np.uint8)
    image_labels = np.array(image_labels)

    # Format checking
    print('image shape : {}'.format(images.shape))
    print('labels shape: {}'.format(image_labels.shape))
    print('image labels [0] shape: {}'.format(image_labels[0].shape))

    # Shuffle dataset
    if shuffle:
        np.random.seed(13)
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images, image_labels = images[indices], image_labels[indices]

    # Save int npz file usable by yolo
    np.savez(os.path.join(path,npz_filename), images=images, boxes=image_labels)
    print('Data saved into {}'.format(os.path.join(path,npz_filename)))

########################################################
######################### Main #########################
########################################################

if __name__ == '__main__':
    args = argparser.parse_args()
    _main(args)
