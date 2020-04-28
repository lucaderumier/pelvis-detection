import numpy as np
import matplotlib.pyplot as plt

#######################################################
############### WORK PURPOSED FUNCTIONS ###############
#######################################################

def is_homogeneous(reg,threshold,mean,dev):
	'''Return true if the region is homogeneous to the original image using standard deviation and a certain threshold.'''
	return (np.abs(np.mean(reg)-mean) <= threshold*dev)

def homo_crop(homog,num_of_reg):
	'''Return the number of colums to crop.'''
	homo_count = 0
	to_crop = 0

	while( homo_count < (num_of_reg//5) and to_crop < (num_of_reg//2) ):

		if(homog[to_crop] == True):
			homo_count += 1
		else:
			homo_count = 0

		to_crop += 1

	while(homog[to_crop] == True and to_crop > 0):
		to_crop -= 1

	return to_crop


def col_splitandcrop(image,threshold,num_of_reg):
	'''Splits the image in num_of_reg columns and return a cropped image of the homogeneous regions of the iamge.'''
	mean = np.mean(image)
	dev = np.sqrt(np.var(image))
	im_len = len(image[0,:,:])
	reg_size = im_len // num_of_reg # number of columns per region

	homog = [True] * num_of_reg


	for i in range(num_of_reg):

		if(i == num_of_reg-1):
			reg = image[:,(i*reg_size):,:]
		else:
			reg = image[:,(i*reg_size):((i+1)*reg_size),:]

		homog[i] = is_homogeneous(reg,threshold,mean,dev)


	to_crop_left = homo_crop(homog,num_of_reg)
	homog.reverse()
	to_crop_right = homo_crop(homog,num_of_reg)


	new_index_left = (to_crop_left+1)*reg_size
	new_index_right = (im_len-to_crop_right)*reg_size


	return image[:,new_index_left:new_index_right,:]

def row_splitandcrop(image,threshold,num_of_reg):
	'''Splits the image in num_of_reg columns and return a cropped image of the homogeneous regions of the iamge.'''
	mean = np.mean(image)
	dev = np.sqrt(np.var(image))
	im_len = len(image[:,0,:])
	reg_size = im_len // num_of_reg # number of columns per region

	homog = [True] * num_of_reg


	for i in range(num_of_reg):

		if(i == num_of_reg-1):
			reg = image[(i*reg_size):,:,:]
		else:
			reg = image[(i*reg_size):((i+1)*reg_size),:,:]

		homog[i] = is_homogeneous(reg,threshold,mean,dev)

	to_crop_top = homo_crop(homog,num_of_reg)
	homog.reverse()
	to_crop_bottom = homo_crop(homog,num_of_reg)

	new_index_top = (to_crop_top+1)*reg_size
	new_index_bottom = (im_len-to_crop_bottom)*reg_size

	return image[new_index_top:new_index_bottom,:,:]


########################################################
############## TESTING PURPOSED FUNCTIONS ##############
########################################################

def special_col_splitandcrop(image,threshold,num_of_reg):
	'''Splits the image in num_of_reg columns and return a cropped image of the homogeneous regions of the image.'''
	mean = np.mean(image)
	dev = np.sqrt(np.var(image))
	im_len = len(image[0,:,:])
	reg_size = im_len // num_of_reg # number of columns per region

	homog = [True] * num_of_reg


	for i in range(num_of_reg):

		if(i == num_of_reg-1):
			reg = image[:,(i*reg_size):,:]
		else:
			reg = image[:,(i*reg_size):((i+1)*reg_size),:]

		homog[i] = is_homogeneous(reg,threshold,mean,dev)


	to_crop_left = homo_crop(homog,num_of_reg)
	homog.reverse()
	to_crop_right = homo_crop(homog,num_of_reg)


	new_index_left = (to_crop_left+1)*reg_size
	new_index_right = (im_len-to_crop_right)*reg_size


	return (new_index_left,new_index_right)

def special_row_splitandcrop(image,threshold,num_of_reg):
	'''Splits the image in num_of_reg columns and return a cropped image of the homogeneous regions of the iamge.'''
	mean = np.mean(image)
	dev = np.sqrt(np.var(image))
	im_len = len(image[:,0,:])
	reg_size = im_len // num_of_reg # number of columns per region

	homog = [True] * num_of_reg


	for i in range(num_of_reg):

		if(i == num_of_reg-1):
			reg = image[(i*reg_size):,:,:]
		else:
			reg = image[(i*reg_size):((i+1)*reg_size),:,:]

		homog[i] = is_homogeneous(reg,threshold,mean,dev)

	to_crop_top = homo_crop(homog,num_of_reg)
	homog.reverse()
	to_crop_bottom = homo_crop(homog,num_of_reg)

	new_index_top = (to_crop_top+1)*reg_size
	new_index_bottom = (im_len-to_crop_bottom)*reg_size

	return (new_index_top,new_index_bottom)


def test_threshold_ratio(image,method,slices):
	'''Test the split and crop function for different threshold and plots the cropping ratio [0,1]. 0 being the image uncropped and 1 being the image fully cropped.
	INPUTS : image, the original 3D image. method, 0 for col_splitandcrop, 1 for row_splitandcrop, 2 for both (col then row), 3 for both (row then col). slices, the number of slices/regions to split the image'''
	threshold = 0.00
	y = []
	x = []

	while(threshold < 0.7):
		x.append(threshold)

		if(method == 0):
			im = col_splitandcrop(image,threshold,slices)
		elif(method == 1):
			im = row_splitandcrop(image,threshold,slices)
		elif(method == 2):
			im2 = col_splitandcrop(image,threshold,slices)
			im = row_splitandcrop(im2,threshold,slices)
		elif(method == 3):
			im2 = row_splitandcrop(image,threshold,slices)
			im = col_splitandcrop(im2,threshold,slices)
		else:
			raise ValueError('method is supposed to be an integer between 0 and 3 included.')

		(rows,cols) = np.shape(im[:,:,0])
		ratio = ((192-rows)+(192-cols))/(192+192)
		y.append(ratio)

		threshold += 0.02


	# Graph 1
	plt.plot(x,y,'g-o')
	plt.title('Graph of the cropping ratio after splitandcrop')
	plt.xlabel('Threshold')
	plt.ylabel('Cropping ratio')
	plt.show()
	print()

def test_borders(image,slices):

	col_max = []
	col_min = []

	row_max = []
	row_min = []

	x = []

	threshold = 0.00

	while(threshold < 0.6):
		x.append(threshold)

		(c_min,c_max) = special_col_splitandcrop(image,threshold,slices)
		(r_min,r_max) = special_row_splitandcrop(image,threshold,slices)

		col_max.append(c_max)
		col_min.append(c_min)

		row_max.append(r_max)
		row_min.append(r_min)

		threshold += 0.02

	# Graph 1
	plt.plot(x,col_max,'r-o',label = 'max col')
	plt.plot(x,col_min,'b-o',label = 'min col')
	plt.title('Graph of the max and min columns with respect to the threshold')
	plt.xlabel('Threshold')
	plt.ylabel('Index')
	plt.legend()
	plt.show()
	print()

	# Graph 2
	plt.plot(x,row_max,'g-o',label = 'max row')
	plt.plot(x,row_min,'y-o',label = 'min row')
	plt.title('Graph of the max and min row with respect to the threshold')
	plt.xlabel('Threshold')
	plt.ylabel('Index')
	plt.legend()
	plt.show()
	print()


###############################################################
############## USEFUL FUNCTION FOR VISUALISATION ##############
###############################################################

def get_coordinates(image3D,threshold,slices):
	cropped_im = col_splitandcrop(image3D,threshold,slices)

	(y0,y1) = special_col_splitandcrop(image3D,threshold,slices) # Left and right indexes
	(x0,x1) = special_row_splitandcrop(cropped_im,threshold,slices) # Top and bottom indexes

	return [y0,y1,x0,x1]

def get_threshold():
	return threshold


image3D = np.load('data/image.npy')


threshold = 0.1


cropped_im = col_splitandcrop(image3D,threshold,100)
cropped_im_final = row_splitandcrop(cropped_im,threshold,100)


np.save('data/cropped_image.npy',cropped_im_final)
