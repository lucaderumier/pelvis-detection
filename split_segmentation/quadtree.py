import numpy as np 

class Node():
	def __init__(self, topleft, side, parent, index, image):
		self.topleft = topleft
		self.side = side
		self.parent = parent
		self.index = index
		self.image = image
		self.mean = np.mean(image)
		self.std = np.std(image)


class Region():
	def __init__(self,init_node):
		self.nodes = [init_node]

		self.pixels = init_node.image.flatten()

		self.sum = np.sum(init_node.image)
		self.size = init_node.image.size

		self.mean = np.mean(init_node.image)
		self.dev = np.std(init_node.image)
		

	def add_node(self,node):
		'''Adds a node to the list and recompute mean/standard deviation for the region.'''
		self.nodes.append(node)

		self.pixels = np.append(self.pixels,node.image.flatten())

		self.sum += np.sum(node.image)
		self.size += node.image.size

		self.mean = self.sum / self.size

		var = np.sum((self.pixels-self.mean)**2)/self.size
		self.dev = np.sqrt(var)




		

		




