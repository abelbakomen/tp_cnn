import numpy as np


class Conv2d:
	# A Convolutional layer using num_filters «kernel_size*kernel_size» filters. 
	def __init__(self,kernel_size=3,num_filters=8):
		self.num_filters = num_filters
		self.kernel_size=kernel_size
		
		#we randomly initialze filters.
		self.filters = np.random.randn(num_filters,kernel_size,kernel_size)/(kernel_size*kernel_size)
	
	def iterate_regions(self, image):
		'''
		Generates all possible kernel_size x kernel_size image regions using valid padding.
		- image is a 2d numpy array
		'''
		h, w = image.shape	
		
		if(min(h,w)<self.kernel_size):
			raise Exception("image width or hieght is too small than the filter")
		
		for i in range(h + 1 - self.kernel_size):
			for j in range(w + 1 - self.kernel_size):
				im_region=image[i:(i+self.kernel_size), j:(j+self.kernel_size)]
				yield im_region, i, j
				
	def forward(self,input):
		'''
			Performs a forward pass of the conv layer using the given input.
			Returns a 3d numpy array with dimensions (h, w, num_filters).
			- input is a 2d numpy array
		'''
		h, w = input.shape
		output = np.zeros((h + 1 - self.kernel_size, w + 1 - self.kernel_size, self.num_filters))
		
		for im_region, i, j in self.iterate_regions(input):
			output[i, j] = np.sum(im_region * self.filters, axis=(1, 2))

		return output
		
