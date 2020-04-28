import numpy as np
from quadtree import Node
from quadtree import Region

def is_homogeneous(reg,threshold,mean,dev):
	'''Return true if the region is homogeneous to the original image using standard deviation and a certain threshold.'''
	return (np.abs(np.mean(reg)-mean) <= threshold*dev)

def unsplittable(node):
	'''Returns true if the node has an odd side length.'''
	return (node.side%2) != 0

def mergeable(node1,node2,threshold):
	'''Returns true if two nodes are neighbours and homogeneous thus mergeable.'''

	# Checks homogeneity
	if(not is_homogeneous(node2.image,threshold,node1.mean,node1.std)):
		return False

	# Checks proximity
	n_1x = (node1.topleft[0],node1.topleft[0]+node1.side) # Node1 X0 - X1
	n_1y = (node1.topleft[1],node1.topleft[1]+node1.side) # Node1 Y0 - Y1
	n_2x = (node2.topleft[0],node2.topleft[0]+node2.side) # Node2 X0 - X1
	n_2y = (node2.topleft[1],node2.topleft[1]+node2.side) # Node2 Y0 - Y1



	if((n_1x[0] <= n_2x[0] and n_1x[1] >= n_2x[1]) or (n_2x[0] <= n_1x[0] and n_2x[1] >= n_1x[1])):
		for y1 in n_1y:
			for y2 in n_2y:
				if(y1 == y2):
					return True

	if((n_1y[0] <= n_2y[0] and n_1y[1] >= n_2y[1]) or (n_2y[0] <= n_1y[0] and n_2y[1] >= n_1y[1])):
		for x1 in n_1x:
			for x2 in n_2x:
				if(x1 == x2):
					return True


	return False





def split(node,last_index):

	if(unsplittable(node)):
		raise Error('This node cannot be split.')

	image = node.image
	side = node.side
	childs = []

	x0 = [0,side//2,0,side//2]
	x1 = [side//2,side,side//2,side]
	y0 = [0,0,side//2,side//2]
	y1 = [side//2,side//2,side,side]

	parent_tl = node.topleft

	tl = [	(parent_tl[0],parent_tl[1]),
			(parent_tl[0]+(side//2),parent_tl[1]),
			(parent_tl[0],parent_tl[1]+(side//2)),
			(parent_tl[0]+(side//2),parent_tl[1]+(side//2))]

	for n in range(4):
		childs.append(Node(tl[n],side//2, node, last_index+1, image[x0[n]:x1[n],y0[n]:y1[n],:]))
		last_index += 1


	return (childs,last_index)

def merge(regions,image,method='Average'):
	'''Takes all the regions one by one and set all the pixels from thoses regions to be the average value of that region.'''
	value = 0
	for reg in regions:

		if(method == 'Average'):
			value = int(reg.mean)
		elif(method == 'Increment'):
			value += 25
		elif(method== 'Exponent'):
			value = int(reg.mean)**2


		for node in reg.nodes:
			tl = node.topleft
			s = node.side
			image[tl[0]:tl[0]+s,tl[1]:tl[1]+s,:] = value

	return image



def splitandmerge(image,threshold,method='Average'):


	root = Node((0,0),len(image[0,:,0]), None, 1, image)
	last_index = 1

	processList = []
	regionList = []

	(child,li) = split(root,last_index)
	processList.extend(child)
	last_index = li


	#################
	# SPLIT SECTION #
	#################

	max_index = 0

	while(processList):
		processElem = processList.pop(0)

		mean = processElem.parent.mean
		dev = processElem.parent.std


		if(is_homogeneous(processElem.image,threshold,mean,dev) or unsplittable(processElem)):
			regionList.append(processElem)
		else:
			(child,li) = split(processElem,last_index)
			processList.extend(child)
			last_index = li


	# From here processList is empty and RegionList contains all the regions to merge

	#################
	# MERGE SECTION #
	#################

	finalRegions = []


	while(regionList):
		firstReg = regionList.pop(0)
		region = Region(firstReg)


		# looking for a similar region
		for toMergeElem in regionList:

			if(mergeable(firstReg,toMergeElem,threshold)):
				region.add_node(toMergeElem)
				regionList.remove(toMergeElem)

		finalRegions.append(region)

	return merge(finalRegions,image,method=method)


image3D = np.load('data/image.npy')

threshold = 0.172

merged_im = splitandmerge(image3D,threshold,method='Average')
np.save('data/merged_image.npy',merged_im)
