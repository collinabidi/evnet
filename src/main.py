from matplotlib import pyplot as plt
from population import Population
from helpers import augment_data
from load_cifar_10 import load_cifar10_data, load_cifar100_data
import numpy as np
import dask.array as da

img_rows, img_cols = 32, 32 # Resolution of inputs
channel = 3 # rgb
num_classes = 10 # cifar 10

# lenet model for testing
"""
conv1 = {'name':'conv1','type':'Convolution2D','border_mode':'same','nb_filter':20,'nb_row':5,'nb_col':5,'input_shape':(img_rows,img_cols,channel)}
activation1 = {'name':'activation1','type':'Activation','activation':'relu'}
max1 = {'name':'max1','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
conv2 = {'name':'conv2','type':'Convolution2D','border_mode':'same','nb_filter':50,'nb_row':5,'nb_col':5}
activation2 = {'name':'activation2','type':'Activation','activation':'relu'}
max2 = {'name':'max2','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
flatten1 = {'name':'flatten1','type':'Flatten'}
dense1 = {'name':'dense2','type':'Dense','output_dim':500}
activation3 = {'name':'activation3','type':'Activation','activation':'relu'}
dense2 = {'name':'dense2','type':'Dense','output_dim':num_classes}
activation4 = {'name':'output','type':'Activation','activation':'softmax'}
"""

#stage 1
conv1_1 = {'name':'conv1_1','type':'Convolution2D','border_mode':'same','nb_filter':64,'nb_row':224,'nb_col':224,'input_shape':(img_rows,img_cols,channel)}
activation1_1 = {'name':'activation1_1','type':'Activation','activation':'relu'}
conv1_2 = {'name':'conv1_2','type':'Convolution2D','border_mode':'same','nb_filter':64,'nb_row':224,'nb_col':224}
activation1_2 = {'name':'activation1_2','type':'Activation','activation':'relu'}
max1 = {'name':'max1','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
#stage 2
conv2_1 = {'name':'conv2_1','type':'Convolution2D','border_mode':'same','nb_filter':112,'nb_row': 128,'nb_col': 128}
activation2_1 = {'name':'activation2_1','type':'Activation','activation':'relu'}
conv2_2 = {'name':'conv2_2','type':'Convolution2D','border_mode':'same','nb_filter':112,'nb_row':128,'nb_col':128}
activation2_2 = {'name':'activation2_2','type':'Activation','activation':'relu'}
max2 = {'name':'max2','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
#stage 3
conv3_1 = {'name':'conv3_1','type':'Convolution2D','border_mode':'same','nb_filter':256,'nb_row': 56,'nb_col': 56}
activation3_1 = {'name':'activation3_1','type':'Activation','activation':'relu'}
conv3_2 = {'name':'conv3_2','type':'Convolution2D','border_mode':'same','nb_filter':256,'nb_row':56,'nb_col':56}
activation3_2 = {'name':'activation3_2','type':'Activation','activation':'relu'}
conv3_3 = {'name':'conv3_3','type':'Convolution2D','border_mode':'same','nb_filter':256,'nb_row':56,'nb_col':56}
activation3_3 = {'name':'activation3_3','type':'Activation','activation':'relu'}
max3 = {'name':'max3','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
#stage 4
conv4_1 = {'name':'conv4_1','type':'Convolution2D','border_mode':'same','nb_filter':512,'nb_row': 28,'nb_col':28}
activation4_1 = {'name':'activation4_1','type':'Activation','activation':'relu'}
conv4_2 = {'name':'conv4_2','type':'Convolution2D','border_mode':'same','nb_filter':512,'nb_row':28,'nb_col':28}
activation4_2 = {'name':'activation4_2','type':'Activation','activation':'relu'}
conv4_3 = {'name':'conv4_3','type':'Convolution2D','border_mode':'same','nb_filter':512,'nb_row':28,'nb_col':28}
activation4_3 = {'name':'activation4_3','type':'Activation','activation':'relu'}
max4 = {'name':'max4','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
#stage 5
conv5_1 = {'name':'conv5_1','type':'Convolution2D','border_mode':'same','nb_filter':512,'nb_row': 14,'nb_col':14}
activation5_1 = {'name':'activation5_1','type':'Activation','activation':'relu'}
conv5_2 = {'name':'conv5_2','type':'Convolution2D','border_mode':'same','nb_filter':512,'nb_row':14,'nb_col':14}
activation5_2 = {'name':'activation5_2','type':'Activation','activation':'relu'}
conv5_3 = {'name':'conv5_3','type':'Convolution2D','border_mode':'same','nb_filter':512,'nb_row':14,'nb_col':14}
activation5_3 = {'name':'activation5_3','type':'Activation','activation':'relu'}
max5 = {'name':'max5','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
#flatten then fully connected (dense) layers
flatten1 = {'name':'flatten1','type':'Flatten'}
dense1 = {'name':'dense1','type':'Dense','output_dim':4096}
activation_dense1 = {'name':'activation_dense1','type':'Activation','activation':'relu'}
dense2 = {'name':'dense2','type':'Dense','output_dim':4096}
activation_dense2 = {'name':'activation_dense2','type':'Activation','activation':'relu'}
dense3 = {'name':'dense3','type':'Dense','output_dim':num_classes}
activation_dense3 = {'name':'output','type':'Activation','activation':'softmax'}
# create population
p = [
	conv1_1,activation1_1,conv1_2,activation1_2,max1,
	conv2_1,activation2_1,conv2_2,activation2_2,max2,
	conv3_1,activation1_1,conv3_2,activation3_2,max3,
	conv4_1,activation1_1,conv4_2,activation4_2,max4,
	conv5_1,activation1_1,conv5_2,activation5_2,max5,
	flatten1,dense1,activation_dense1,dense2,activation_dense2,dense3,activation_dense3
	]
pop = Population(p,size=5)
# Example to fine-tune on samples from Cifar10
batch_size = 64
nb_epoch = 12
train_num = 1000
valid_num = 1000
X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols, nb_train_samples=train_num,nb_valid_samples=valid_num)
X_train,X_valid = X_train.astype('float32'), X_valid.astype('float32')

augment_ratio = 2
X_train, Y_train = augment_data(X_train,Y_train,batch_size,augment_ratio)
"""
X_train = da.from_array(np.asarray(X_train), chunks=(int(train_num/augment_ratio))*augment_ratio)
Y_train = da.from_array(np.asarray(Y_train), chunks=(int(train_num/augment_ratio))*augment_ratio)
X_valid = da.from_array(np.asarray(X_valid), chunks=(int(valid_num/augment_ratio))*augment_ratio)
Y_valid = da.from_array(np.asarray(Y_valid), chunks=(int(valid_num/augment_ratio))*augment_ratio)
"""
# run train and evalute
pop.train_evaluate_population(X_train,Y_train,batch_size,nb_epoch,X_valid,Y_valid)