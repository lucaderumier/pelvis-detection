import numpy as np
from skimage.color import gray2rgb
import matplotlib.pyplot as plt
from splitandcrop import get_coordinates


def draw_bounding_box(image,bottom,up,left,right,width,height,color):
    '''Draw the resulting bouding box. xy is the bottom left coordiante of the box.'''
    output = image
    output = gray2rgb(output)

    output[up,left:(left+width)] = color
    output[bottom,left:(left+width)] = color
    output[up:(up+height),left] = color
    output[up:(up+height),right] = color

    return output


def show_slices_gen(image, rows, cols, info, filename,y0,y1,x0,x1,color):
    # INPUTS: image of size (rows,cols,160), info is the message to be displayed above the image, filename is the name of the file to be saved (or None is the image must not be saved)
    im = np.zeros((rows*10,cols*16,3))
    plt.figure(figsize=(40,20))

    # BOUNDING BOX
    width = y1-y0
    height = x1-x0

    for s in range(160):
        output = draw_bounding_box(image[:,:,s],x1,x0,y0,y1,width,height,color)
        line = (s//16)*rows
        col = (s%16)*cols
        im[line:(line+rows),col:(col+cols),:] = output

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


image = np.load('../show_images/data/CHA-CBCT-2-image.npy') 
[y0,y1,x0,x1] = get_coordinates(image,0.1,100)

info = 'patient 0'
filename = None
(rows,cols) = np.shape(image[:,:,0])

red = np.array([255,0,0])


show_slices_gen(image,rows,cols, info, filename,y0,y1,x0,x1,red)
