import numpy as np
from skimage.color import gray2rgb
import matplotlib.pyplot as plt

def show_slices_gen(image, rows, cols, info, filename):
    # INPUTS: image of size (rows,cols,160), info is the message to be displayed above the image, filename is the name of the file to be saved (or None is the image must not be saved)
    im = np.zeros((rows*10,cols*16,3))
    plt.figure(figsize=(40,20))

    for s in range(160):
        output = gray2rgb(image[:,:,s])
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

#image = np.load('data/cropped_image.npy') # CROP
image = np.load('data/merged_image.npy') # MERGE

info = 'patient 0'
filename = None
(rows,cols) = np.shape(image[:,:,0])

show_slices_gen(image,rows,cols, info, filename)
