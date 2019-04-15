import numpy as np
import dask.array as da
import tensorflow as tf

from matplotlib import pyplot as plt
from population import Population
from helpers import augment_data
from load_cifar_10 import load_cifar10_data, load_cifar100_data
from helpers import plot_history

if __name__ == "__main__":
	img_rows, img_cols = 32, 32 # Resolution of inputs
	channel = 3 # rgb
	num_classes = 10 # cifar 100

	# lenet model for testing
	conv1 = {'name':'conv1','type':'Convolution2D','border_mode':'same','nb_filter':20,'nb_row':5,'nb_col':5,'activation':'relu','input_shape':(img_rows,img_cols,channel)}
	max1 = {'name':'max1','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
	conv2 = {'name':'conv2','type':'Convolution2D','border_mode':'same','nb_filter':50,'nb_row':5,'nb_col':5,'activation':'relu'}
	max2 = {'name':'max2','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
	flatten1 = {'name':'flatten1','type':'Flatten'}
	dense1 = {'name':'dense2','type':'Dense','output_dim':500,'activation':'relu'}
	dense2 = {'name':'dense2','type':'Dense','output_dim':num_classes,'activation':'softmax'}
	p = [conv1,max1,conv2,max2,flatten1,dense1,dense2]


	pop = Population(p,size=5,k_best=2)

	batch_size = 16
	nb_epoch = 25
	train_num = 5000
	test_num = 1000

	X_train, Y_train, X_test, Y_test = load_cifar10_data(img_rows, img_cols, nb_train_samples=train_num,nb_test_samples=test_num)
	augment_ratio = 5
	X_train, Y_train = augment_data(X_train,Y_train,batch_size,augment_ratio)

	generations = 3

	for i in range(0,generations):
		# run train and evalute
		pop.train_evaluate_population(X_train,Y_train,batch_size,nb_epoch,X_test,Y_test)
		pop.evolve()

	for gen in range(0,generations):
		plot_history(pop.generation_histories[gen],nb_epoch)
