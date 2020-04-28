
import numpy as np
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from scipy.misc import imfilter
from splitandcrop import get_coordinates
from splitandcrop import get_threshold


def image_with_contours_gen(image, bm1, bm2, color1, color2,bottom,up,left,right,width,height,box_color):
    # INPUTS: image of size (192,192,160), bm1 is the first set of contours (192,192,160,n_contours), bm2 is the second set of contours, col1 is the first set of colors, col2 is the second set of colors
    output = image
    output = gray2rgb(output)
    n_contours = bm1.shape[2]
    for contour_num in range(n_contours):
        edges = imfilter(bm1[:,:,contour_num]*255, 'find_edges')
        output[edges>0] = np.array(color1[contour_num,:])
    for contour_num in range(n_contours):
        edges = imfilter(bm2[:,:,contour_num]*255, 'find_edges')
        output[edges>0] = np.array(color2[contour_num,:])

    output[up,left:(left+width)] = box_color
    output[bottom,left:(left+width)] = box_color
    output[up:(up+height),left] = box_color
    output[up:(up+height),right] = box_color

    return output

def show_slices_gen(image, bm1, bm2, col1, col2, info, filename,y0,y1,x0,x1,color):
    # INPUTS: image of size (192,192,160), bm1 is the first set of contours (192,192,160,n_contours), bm2 is the second set of contours, col1 is the set of colors for bm1, col2 is the set of colors for bm2, info is the message to be displayed above the image, filename is the name of the file to be saved (or None is the image must not be saved)
    im = np.zeros((192*10,192*16,3))
    plt.figure(figsize=(40,20))

    # BOUNDING BOX
    width = y1-y0
    height = x1-x0


    for s in range(160):
        output = image_with_contours_gen(image[:,:,s], bm1[:,:,s,:], bm2[:,:,s,:], col1, col2,x1,x0,y0,y1,width,height,color)
        line = (s//16)*192
        col = (s%16)*192
        im[line:(line+192),col:(col+192),:] = output
    im = ((im - np.min(im)) * 255 / (np.max(im) - np.min(im))).astype(np.uint8)
    frame1 = plt.gca()
    frame1.axes.get_xaxis().set_visible(False)
    frame1.axes.get_yaxis().set_visible(False)
    plt.imshow(im)
    print(info)
    plt.show()
    if filename != None:
        plt.imsave(filename, im, dpi=1000)
    plt.close()

image = np.load('data/merged_image.npy') # MERGE
mask = np.load('data/mask.npy')
mask = mask[:,:,:,:-1]
pred = np.zeros((192,192,160,2))
col1 = np.array([[255,0,0],
                     [0,128,0],
                     [0,0,255]])
col2 = np.array([[255,0,255],
                     [0,255,0],
                     [0,255,255]])


[y0,y1,x0,x1] = get_coordinates(image,get_threshold(),100)
info = 'patient 0'
filename = None
blue = np.array([0,0,255])
show_slices_gen(image, mask, pred, col1, col2, info, filename,y0,y1,x0,x1,blue)
