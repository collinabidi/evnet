import numpy as np
import dask.array as da
import tensorflow as tf
import os
import keras 

from matplotlib import pyplot as plt
from population import Population
from load_cifar_10 import load_cifar10_data, load_cifar100_data
from helpers import plot_history
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10,mnist
from keras import backend as K

# get current working directory 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cwd = os.getcwd()

if __name__ == "__main__":
	img_rows, img_cols = 32, 32 # Resolution of inputs
	channel = 1 # rgb
	num_classes = 10 # cifar 10

	# lenet model for testing
	conv1 = {'name':'conv1','type':'Convolution2D','border_mode':'same','nb_filter':6,'nb_row':3,'nb_col':3,'activation':'relu','input_shape':(img_rows,img_cols,channel)}
	max1 = {'name':'max1','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
	conv2 = {'name':'conv2','type':'Convolution2D','border_mode':'same','nb_filter':16,'nb_row':3,'nb_col':3,'activation':'relu'}
	max2 = {'name':'max2','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
	flatten1 = {'name':'flatten1','type':'Flatten'}
	dense1 = {'name':'dense2','type':'Dense','output_dim':120,'activation':'relu'}
	dense2 = {'name':'dense2','type':'Dense','output_dim':84,'activation':'relu'}
	dense3 = {'name':'dense2','type':'Dense','output_dim':num_classes,'activation':'softmax'}
	p = [conv1,max1,conv2,max2,flatten1,dense1,dense2,dense3]


	pop = Population(p,size=3,k_best=2)

	batch_size = 128
	nb_epoch = 2
	train_num = 50000
	test_num = 1000
	
	x_train, y_train, x_test, y_test = load_cifar10_data(img_rows, img_cols, nb_train_samples=train_num,nb_test_samples=test_num)
	x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.2)
	'''
	(x_train, y_train), (x_test, y_test) = mnist.load_data()

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, img_rows, img_cols)
		x_test = x_test.reshape(x_test.shape[0], 1, img_rows, img_cols)
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)
		input_shape = (img_rows, img_cols, 1)
	'''
	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255
	print('x_train shape:', x_train.shape)
	print(x_train.shape[0], 'train samples')
	print(x_test.shape[0], 'test samples')
	x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.2)
	
	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, num_classes)
	y_test = keras.utils.to_categorical(y_test, num_classes)

	print("%s %s %s %s" % (str(x_train.shape),str(x_test.shape),str(y_train.shape),str(y_test.shape)))

	generations = 1

	for i in range(0,generations):
		# run train and evalute
		if i > 0:
			pop.evolve()
		big_poppa = pop.train_evaluate_population(x_train,y_train,x_valid,y_valid,batch_size,nb_epoch,x_test,y_test)

	for gen in range(0,generations):
		plot_history(pop.generation_histories[gen],nb_epoch)

	print("TYPE: " + str(type(big_poppa)))
	print ("Final winner : " + big_poppa.name)
	
	# CONVERT AND SAVE TO TFLITE FILE
	keras_file = str(big_poppa.name)+".h5"
	converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file("models/"+keras_file)
	tflite_model = converter.convert()
	open("/models/winner_"+str(individual.name)+".tflite", "wb").write(tflite_model)