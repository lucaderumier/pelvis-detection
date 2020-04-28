import numpy as np
import os
import pickle
import PIL.Image
from skimage.color import gray2rgb

def load_annotation(annot_path):
    'Loads annot_file into a dictionnary.'

    with open(annot_path, 'rb') as fp:
        data = pickle.load(fp)

    return data

def yolo_dataset(dict,path,npz_filename,label_dict,shuffle = False):
    'Transform our dictionnary and images to npz data file that yolo can use.'
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
                if(organ in dict[filename]['bb'].keys()):
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
    print('IMAGE SIZE : {}'.format(images.shape))
    print('IMAGE LABELS SIZE : {}'.format(image_labels.shape))
    print('IMAGE LABELS[0] SIZE : {}'.format(image_labels[0].shape))

    # Shuffle dataset
    if shuffle:
        np.random.seed(13)
        indices = np.arange(len(images))
        np.random.shuffle(indices)
        images, image_labels = images[indices], image_labels[indices]

    # Save int npz file usable by yolo
    np.savez(os.path.join(path,npz_filename), images=images, boxes=image_labels)
    print('Data saved into {}'.format(os.path.join(path,npz_filename)))


if __name__ == '__main__':
    # Dictionnary of the organs and their index in the classes.txt file
    #label_dict = {'bladder' : 0, 'rectum' : 1, 'prostate' : 2} # CHANGE THIS
    #DATA_SET = 'DATA_CT_2' # CHANGE THIS

    # Environment variables
    SETS = ['train','val','test']
    for SET in SETS:
        DATASET_DIR = os.path.join('pelvis_scan',DATA_SET,SET)
        ANNOTATION = 'annotations_'+SET+'.p'
        FILENAME = 'pelvis_data_'+SET+'.npz'
        dict = load_annotation(os.path.join(DATASET_DIR,ANNOTATION))
        yolo_dataset(dict,DATASET_DIR,FILENAME,label_dict)
