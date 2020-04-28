
import numpy as np
import matplotlib.pyplot as plt
import os
from splitandcrop import get_coordinates, get_threshold
from splitandmerge import splitandmerge


###############################################################################
#############################   USEFUL FUNCTIONS  #############################
###############################################################################

def perfect_box_coord(mask):
	'''Input: a segmentation mask. Ouput : the perfect bounding box coordinates.'''
	bladder = mask[:,:,:,0]
	rectum = mask[:,:,:,1]

	m = mask.shape[0]
	n = mask.shape[1]

	# starting with inverse position to not get stuck if there is no mask in the image
	top = m
	bottom = 0
	left = n
	right = 0

	# iterating on the depth
	for s in range(mask.shape[2]):
		for i in range(m):
			for j in range(n):

				if(bladder[i,j,s]): 
					if(i < top): 	top    = i
					if(i > bottom): bottom = i
					if(j < left):	left   = j
					if(j > right):  right  = j

				if(rectum[i,j,s]): 
					if(i < top): 	top    = i
					if(i > bottom): bottom = i
					if(j < left):	left   = j
					if(j > right):  right  = j
	

	return [left,right,top,bottom]

def total_voxels(image):
	''' Input: image
		Output: total number of voxels in the image'''
	return image.shape[0]*image.shape[1]

#####################################################################
#############################   TP|TN   #############################
#############################   FP|FN   #############################
############################# FUNCTIONS #############################
#####################################################################

def true_positif(image,mask,threshold,threshold_merge = 0.172,method = 'crop',slices = 100):
	'''Input: an image, the corresponding mask, a method, a threshold and the number of slices. 
	   Ouput : the number of voxels that are in our bounding box and in the perfect box.'''
	if(method == 'crop'):
		[left,right,top,bottom] = get_coordinates(image,threshold,slices)
	elif(method == 'merge'):
		image = splitandmerge(image,threshold_merge,method='Exponent')
		[left,right,top,bottom] = get_coordinates(image,threshold,slices)
	else:
		raise Exception('Unknown method')

	[p_left,p_right,p_top,p_bottom] = perfect_box_coord(mask)

	tp_top = 0
	tp_bottom = 0
	tp_left = 0
	tp_right = 0

	total_area = (bottom-top)*(left-right)

	if(p_left>left and p_left<right):       tp_left = p_left
	elif(p_left<left and left<p_right):     tp_left = left

	if(p_right<right and p_right>left):     tp_right = p_right
	elif(p_right>right and right>p_left):   tp_right = right

	if(p_top>top and p_top<bottom):         tp_top = p_top
	elif(p_top<top and top<p_bottom):       tp_top = top

	if(p_bottom<bottom and p_bottom>top):   tp_bottom = p_bottom
	elif(p_bottom>bottom and bottom>p_top): tp_bottom = bottom


	return (tp_right-tp_left)*(tp_bottom-tp_top)


def true_negatif(image,mask,threshold,threshold_merge = 0.172,method = 'crop',slices = 100):
	'''Input: an image, the corresponding mask, a method, a threshold and the number of slices. 
	   Ouput : the number of voxels that are in our bounding box but shouldnt.'''
	tp = true_positif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices =slices)

	[left,right,top,bottom] = get_coordinates(image,threshold,slices)
	total_bb = (right-left)*(bottom-top)

	return total_bb-tp

def false_positif(image,mask,threshold,threshold_merge = 0.172,method = 'crop',slices = 100):
	'''Input: an image, the corresponding mask, a method, a threshold and the number of slices. 
	   Ouput : the number of voxels that are not in our bounding box and shouldnt be.'''
	tp = true_positif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices =slices)
	total = total_voxels(image)

	[left,right,top,bottom] = get_coordinates(image,threshold,slices)
	total_bb = (right-left)*(bottom-top)

	[p_left,p_right,p_top,p_bottom] = perfect_box_coord(mask)
	total_pb = (p_right-p_left)*(p_bottom-p_top)

	return total-total_bb-total_pb+tp

def false_negatif(image,mask,threshold,threshold_merge = 0.172,method = 'crop',slices = 100):
	'''Input: an image, the corresponding mask, a method, a threshold and the number of slices. 
	   Ouput : the number of voxels that are not in our bounding box and should be.'''
	tp = true_positif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices =slices)

	[p_left,p_right,p_top,p_bottom] = perfect_box_coord(mask)
	total_pb = (p_right-p_left)*(p_bottom-p_top)

	return total_pb-tp

def precision(image,mask,threshold,threshold_merge = 0.172,method = 'crop',slices = 100):
	'''Input: an image, the corresponding mask, a method, a threshold and the number of slices. 
	   Ouput : the precision of the algorithm.'''

	tp = true_positif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices =slices)
	fp = false_positif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices =slices)

	return tp/(tp+fp)

def recall(image,mask,threshold,threshold_merge = 0.172,method = 'crop',slices = 100):
	'''Input: an image, the corresponding mask, a method, a threshold and the number of slices. 
	   Ouput : the precision of the algorithm.'''

	tp = true_positif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices =slices)
	fn = false_negatif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices =slices)

	return tp/(tp+fn)

def average_test(threshold,test,threshold_merge = 0.172,method = 'crop',slices = 100):
	'''Input: threshold and number of slices. 
	   Output: average true positif'''

	print('ENTERING AVERAGE ' + str(test) + ' TEST WITH THRESHOLD ' + str(threshold))


	average = 0
	count  = 0

	path = '../show_images/data/'
	dir_list = sorted(os.listdir(path))

	for filename in dir_list:
		

		if(count%2==1 and filename.endswith(".npy")):
			print(path+filename)
			image = np.load(path+filename)

		elif(count%2==0 and filename.endswith(".npy")):
			print(path+filename)
			mask = np.load(path+filename)

			if(test == 'TRUE POSITIF'):
				average += true_positif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices = slices)
			elif(test == 'TRUE NEGATIF'):
				average += true_negatif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices = slices)
			elif(test == 'FALSE POSITIF'):
				average += false_positif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices = slices)
			elif(test == 'FALSE NEGATIF'):
				average += false_negatif(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices = slices)
			elif(test == 'PRECISION'):
				average += precision(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices = slices)
			elif(test == 'RECALL'):
				average += recall(image,mask,threshold,threshold_merge = threshold_merge,method = method,slices = slices)

		count += 1

	n = (count/2)

	print()

	return average/n

#########################################################################
#############################  OTHER TESTS  #############################
#########################################################################

def percentage_outside_box_crop(image,mask,threshold,threshold_merge = 0.172,method = 'crop',slices = 100):
	'''Input: an image, the corresponding mask, a method, a threshold and the number of slices. 
	   Ouput: percentage of the organs that our outside the box.'''
	if(method == 'crop'):
		[left,right,top,bottom] = get_coordinates(image,threshold,slices)
	elif(method == 'merge'):
		image = splitandmerge(image,threshold_merge,method='Exponent')
		[left,right,top,bottom] = get_coordinates(image,threshold,slices)
	else:
		raise Exception('Unknown method')

	[p_left,p_right,p_top,p_bottom] = perfect_box_coord(mask)

	h_top = 0
	h_bottom = 0
	w_left = 0
	w_right = 0

	total_area = (bottom-top)*(left-right)

	if(p_left<left):     w_left  = left-p_left
	if(p_right>right):   w_right = p_right-right
	if(p_top<top): 	     h_top   = top-p_top
	if(p_bottom>bottom): h_bottom  = p_bottom-bottom

	h = h_top+h_bottom
	w = w_left+w_right

	if(h == 0 and w == 0):
		return 0
	elif(h == 0):
		h =  bottom-top
	elif(w == 0):
		w = left-right

	return ((h*w)/total_area)*100

def average_percentage_outside_box(threshold,method = 'crop',slices = 100):
	'''Input: threshold and number of slices. 
	   Output: average percentage of the area of the perfect box that's outisde our bouding box.'''

	print('ENTERING AVERAGE OUT PERCENTAGE TEST WITH THRESHOLD ' + str(threshold))

	average = 0
	count  = 0

	path = '../show_images/data/'
	dir_list = sorted(os.listdir(path))

	for filename in dir_list:
		print(path+filename)

		if(count%2==1 and filename.endswith(".npy")):
			image = np.load(path+filename)

		elif(count%2==0 and filename.endswith(".npy")):
			mask = np.load(path+filename)

			average += percentage_outside_box_crop(image,mask,threshold,method = method,slices = slices)

		count += 1

	n = (count/2)

	return average/n

def percentage_cropped(image,threshold,threshold_merge = 0.172,method = 'crop',slices = 100):
	'''Input: an image, a threshold, a method and a number of slices. 
	   Ouput: the percentage of the image that has been cropped.'''
	if(method == 'crop'):
		[left,right,top,bottom] = get_coordinates(image,threshold,slices)
	elif(method == 'merge'):
		image = splitandmerge(image,threshold_merge,method='Exponent')
		[left,right,top,bottom] = get_coordinates(image,threshold,slices)
	else:
		raise Exception('Unknown method')


	(rows,cols) = (image.shape[0],image.shape[1])
	(box_rows,box_cols) = (bottom-top,right-left)

	return (1-((box_rows*box_cols)/(rows*cols)))*100

def average_percentage_cropped(threshold,method = 'crop',slices = 100):

	print('ENTERING AVERAGE CROPPED PERCENTAGE TEST WITH THRESHOLD ' + str(threshold))

	average = 0
	n = 0

	path = '../show_images/data/'
	dir_list = sorted(os.listdir(path))

	for filename in dir_list:

		if(filename.endswith("image.npy")):
			image = np.load(path+filename)
			print(path+filename)
			average += percentage_cropped(image,threshold,method = method,slices = slices)
			n += 1
			

	return average / n


if __name__ == "__main__":

	threshold = 0.1
	
	########## AVERAGE PERCENT OUTSIDE THE BOX ##########
	'''
	[out_t005,out_t01,out_t02,out_t03,out_t04,out_t05,out_t07,out_t09] = [average_percentage_outside_box(0.05),average_percentage_outside_box(0.1),average_percentage_outside_box(0.2),average_percentage_outside_box(0.3),average_percentage_outside_box(0.4),average_percentage_outside_box(0.5),average_percentage_outside_box(0.7),average_percentage_outside_box(0.9)]
	print(str([out_t005,out_t01,out_t02,out_t03,out_t04,out_t05,out_t07,out_t09]))
	'''
	########## TEST SUITE ##########
	[tp_t005,tp_t01,tp_t02,tp_t03] = [average_test(0.05,'TRUE POSITIF'),average_test(0.1,'TRUE POSITIF'),average_test(0.2,'TRUE POSITIF'),average_test(0.3,'TRUE POSITIF')]
	[tn_t005,tn_t01,tn_t02,tn_t03] = [average_test(0.05,'TRUE NEGATIF'),average_test(0.1,'TRUE NEGATIF'),average_test(0.2,'TRUE NEGATIF'),average_test(0.3,'TRUE NEGATIF')]
	[fp_t005,fp_t01,fp_t02,fp_t03] = [average_test(0.05,'FALSE POSITIF'),average_test(0.1,'FALSE POSITIF'),average_test(0.2,'FALSE POSITIF'),average_test(0.3,'FALSE POSITIF')]
	[fn_t005,fn_t01,fn_t02,fn_t03] = [average_test(0.05,'FALSE NEGATIF'),average_test(0.1,'FALSE NEGATIF'),average_test(0.2,'FALSE NEGATIF'),average_test(0.3,'FALSE NEGATIF')]

	[pre_t005,pre_t01,pre_t02,pre_t03] = [average_test(0.05,'PRECISION'),average_test(0.1,'PRECISION'),average_test(0.2,'PRECISION'),average_test(0.3,'PRECISION')]
	[rec_t005,rec_t01,rec_t02,rec_t03] = [average_test(0.05,'RECALL'),average_test(0.1,'RECALL'),average_test(0.2,'RECALL'),average_test(0.3,'RECALL')]

	print()
	print('######################################################')
	print('##################### RESULTS ########################')
	print('######################################################')
	print()
	print('TRUE POSITIF : ' + str([tp_t005,tp_t01,tp_t02,tp_t03]))
	print('TRUE NEGATIF : ' + str([tn_t005,tn_t01,tn_t02,tn_t03]))
	print('FALSE POSITIF : ' + str([fp_t005,fp_t01,fp_t02,fp_t03]))
	print('FALSE NEGATIF : ' + str([fn_t005,fn_t01,fn_t02,fn_t03]))
	print()
	print('PRECISION : ' + str([pre_t005,pre_t01,pre_t02,pre_t03]))
	print('RECALL : ' + str([rec_t005,rec_t01,rec_t02,rec_t03]))



