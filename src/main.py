import numpy as np
import dask.array as da
import tensorflow as tf

from matplotlib import pyplot as plt
from population import Population
from helpers import augment_data
from load_cifar_10 import load_cifar10_data, load_cifar100_data

img_rows, img_cols = 32, 32 # Resolution of inputs
channel = 3 # rgb
num_classes = 10

# lenet model for testing
conv1 = {'name':'conv1','type':'Convolution2D','border_mode':'same','nb_filter':20,'nb_row':5,'nb_col':5,'input_shape':(img_rows,img_cols,channel)}
act1 = {'name':'act1','type':'Activation','activation':'relu'}
max1 = {'name':'max1','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
conv2 = {'name':'conv2','type':'Convolution2D','border_mode':'same','nb_filter':50,'nb_row':5,'nb_col':5,'activation':'relu'}
act2 = {'name':'act2','type':'Activation','activation':'relu'}

max2 = {'name':'max2','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
flatten1 = {'name':'flatten1','type':'Flatten'}
dense1 = {'name':'dense2','type':'Dense','output_dim':500}
act3 = {'name':'act3','type':'Activation','activation':'relu'}

dense2 = {'name':'dense2','type':'Dense','output_dim':num_classes}
act3 = {'name':'act3','type':'Activation','activation':'softmax'}

p = [conv1,act1,max1,conv2,act2,max2,flatten1,dense1,act3,dense2,act3]


pop = Population(p,size=15,k_best=5)

batch_size = 32
nb_epoch = 15
train_num = 3000
valid_num = 1000
generations = 2

X, Y, X_test, Y_test = load_cifar10_data(img_rows, img_cols, nb_train_samples=train_num,nb_test_samples=valid_num)
augment_ratio = 3
#X, Y = augment_data(X_train,Y_train,len(X_train),4)

for i in range(0,generations):
	# run train and evalute
	pop.train_evaluate_population(X,Y,batch_size,nb_epoch,X_test,Y_test)
	pop.evolve()

for gen in range(0,generations):
	plot_history(pop.generation_histories[gen],nb_epoch)
