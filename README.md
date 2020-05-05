# Organs detection in CT and CBCT using deep learning for radiotherapy applications

## Description

This repository contains the files that were used for my master thesis.
The aim of the methods defined in this repositroy is to detect organs (bladder, rectum, prostate) in CT and CBCT scans of the male pelvic area and extract a boudning box containing all of them. This repositroy contains two distinct classes of algorithms : the classical computer vision based methods (split_segmentaion) and the learning based method (pelvis_yolo). Both are described in below.

--------------------------------------------------------------------------------

## Deep learning based method

This code was adapted from https://github.com/allanzelener/YAD2K and modified to recognize bladder, prostate and rectum on pelvic CT and CBCT scan images.

YAD2K is a 90% Keras/10% Tensorflow implementation of YOLO_v2.

Original paper: [YOLO9000: Better, Faster, Stronger](https://arxiv.org/abs/1612.08242) by Joseph Redmond and Ali Farhadi.


### Requirements

- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [Pillow](https://pillow.readthedocs.io/) (For rendering test results.)
- [Python 3](https://www.python.org/)

### Usage

- [Keras](https://github.com/fchollet/keras)
- [Tensorflow](https://www.tensorflow.org/)
- [Numpy](http://www.numpy.org/)
- [Pillow](https://pillow.readthedocs.io/) (For rendering test results.)
- [Python 3](https://www.python.org/)

--------------------------------------------------------------------------------

## Classical computer vision methods

The split_segmentation folder contains the files for split and merge segmentation and split and crop segmentation methods applied to pelvis Cone Beam CT scans, as well as utility files to plot and analyse the results.


### Usage

To use this code on your own data, start by creating a data folder inside the repository. Then add two files to the data folder : 'image.npy' and 'mask.npy' of shape (192,192,160) and (192,192,160,3). 'mask.npy' contains three segmentation masks (bladder, rectum and prostate) of ones (if the organ belongs to the mask) and zeros (if it does not).

Once this is done, you can run the command line scripts to display the results of the different algorithms on this particular image.

- run_merge.sh : runs the split and merge segmentation method on the image. Then saves the result as 'merged_image.npy' and displays it.
- run_crop.sh : runs the split and crop segmentation method on the image. Then saves the result as 'cropped_image.npy' and displays it (along with the ground truth bounding box containing all the organs).
- run_spamcrop.sh : runs the split and merge segmentation method on the image and feeds the result as input of the split and crop segmentation method. Then saves the result as 'cropped_image.npy' and displays it (along with the ground truth bounding box containing all the organs).

The ground truth bounding box is displayed in blue and the predicted box in purple. A piece of the results you should obtained after running each file is displayed below.

![run_merge.sh](split_segmentation/etc/spam.png){:height="50%" width="50%"}
![run_crop.sh](split_segmentation/etc/crop.png){:height="50%" width="50%"}
![run_spamcrop.sh](split_segmentation/etc/spamcrop.png){:height="50%" width="50%"}
