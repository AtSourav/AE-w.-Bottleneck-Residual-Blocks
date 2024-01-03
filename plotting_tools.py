"""
@author: sourav
"""

"""
Module for plotting functions that I often use in my notebooks on CNN based architectures on image datasets.
"""

import numpy as np
import matplotlib.pyplot as plt


def plot_dataset_images(img_set,fig_size,rows,columns,axis_arg='off'):
	"""
	Plots a sample of images from dataset in question.
	
	Parameters
	..........
	img_set   : A numpy ndarray or a tensor comprising of a batch of
		    images with shape [batch_size,img_ht,img_wd,channels].
		    The image tensors should have values rescaled to 
		    the range [0,1].
		    
	fig_size  : A tuple (x,y) specifying the size of the figure
  		    (comprising of the all the images being plotted). x, 
		     and y are the dimensions along the respective axes.
		     
	rows      : The number of rows of images.
	
	columns   : The number of columns of images.
	
	axis_arg  : The arguments for plt.axis(). The default setting 'off'
		     ensures that the axes are not shown or labelled. For 
		     other options, see docu for matplotliv.pyplot.axis().
		     
		     
	Returns
	.......
	A plot of images from the dataset with as many rows and columns as
	specified.
	
	"""

	visual = plt.figure(figsize=fig_size)

	for image in img_set:            
  		for i in range(rows*columns):
    		visual.add_subplot(rows,columns,i+1)
    		plt.imshow(image)
    		plt.axis(axis_arg)
    		
    		
    		
    		
    		
def plot_reconstruction(set, name, num_rows, img_edge, seed_val, epochs, axis_arg='off'): 

	"""
	Plot a square grid of images from the dataset and the corresponding reconstructed images.
	
	Parameters
	..........
	set         : A numpy ndarray or a tensor containing all the images of a specific batch
		      (training or validation). Image tensors should have pixel values rescaled
		      to the range [0,1].
		      
	name        : 'training' or 'validation' depending on which set of images is chosen.
	
	num_rows    : The number of rows of images to be plotted. We're plotting a square grid of
		      images for the original images, and likewise for the reconstructions. So the 
		      number of images of each kind will be num_rows**2 .
		      
	img_edge    : The edge length of each square image. Will determine the figsize of the
		      entire plot.
		      
	seed_val    : A seed value for selecting images randomly from the chosen set.
	
	epochs      : Number of epochs for which the network has been trained.
	
	axis_arg    : The arguments for plt.axis(). The default setting 'off' ensures that the 
		      axes are not shown or labelled. For other options, see docu for 
		      matplotliv.pyplot.axis() .
		      
		      
	Returns
	.......
	A square grid of num_rows**2 number of original images from the chosen set, and a square 
	grid of the corresponding reconstructed images to it's right. The two grids are separated
	by some clear space. 
	"""    
	
	np.random.seed(seed_val)

	tg_indices = np.random.randint(0,set.shape[0],size = num_rows**2)
	img_sample = tf.convert_to_tensor(np.array(set)[tg_indices])
	z_sample = encoder(img_sample)
	img_recon = decoder(z_sample)
	
	num_col = 2*num_rows + 1

	recon = plt.figure(figsize=(img_edge*num_col,img_edge*num_rows))    
	recon.suptitle('Reconstructed images (right) from the ' + name + '-set after ' + str(epochs) + ' epochs: no regularisation in the encoder, decoder, latent_dim = 512', fontweight= 'bold', y=0.93)

	recon.tight_layout()

	for i in range(num_rows*num_col):
		recon.add_subplot(num_rows,num_col,i+1)
		j = int(np.floor(i/num_col))
		if (i % num_col) < num_rows:
			img = set[tg_indices[j*num_rows + (i % num_col)]]
			plt.imshow(img)
			plt.axis(axis_arg)
		elif (i % num_col) == num_rows:
			img = np.ones((64,64,3))       # the size of this image doesn't really matter
			plt.imshow(img)
			plt.axis(axis_arg)
		else:
			img = img_recon[j*num_rows + (i % num_col) - num_rows - 1]
			plt.imshow(img)
			plt.axis(axis_arg)
