#! python3
# helper functions
import tensorflow as tf
import numpy as np
import gc
import time
import keras

from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.backend.tensorflow_backend import set_session,clear_session,get_session
from keras.datasets import cifar10,cifar100,mnist
from keras import backend as K

def load_mnist_data(img_rows, img_cols, nb_train_samples, nb_test_samples):
	# the data, split between train and test sets
	(x_train, y_train), (x_test, y_test) = mnist.load_data()
	print("****ORIGINAL****\nx_train shape:%s\ny_train shape:%s\nx_test shape:%s\ny_test shape:%s" % (str(x_train.shape),str(y_train.shape),str(x_test.shape),str(y_test.shape)))

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 1, 28, 28)
		x_test = x_test.reshape(x_test.shape[0], 1, 28, 28)
		input_shape = (1, 28, 28)
	else:
		x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
		x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
		input_shape = (28, 28, 1)

	# resize arrays
	x_train = np.resize(x_train,(nb_train_samples,x_train.shape[1],x_train.shape[2],x_train.shape[3]))
	x_test = np.resize(x_test,(nb_test_samples,x_test.shape[1],x_test.shape[2],x_test.shape[3]))
	y_train = np.resize(y_train,(nb_train_samples,))
	y_test = np.resize(y_test,(nb_test_samples),)

	x_train = x_train.astype('float32')
	x_test = x_test.astype('float32')
	x_train /= 255
	x_test /= 255

	# convert class vectors to binary class matrices
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)

	print("****NEW****\nx_train shape:%s\ny_train shape:%s\nx_test shape:%s\ny_test shape:%s" % (str(x_train.shape),str(y_train.shape),str(x_test.shape),str(y_test.shape)))
	return x_train, y_train, x_test, y_test

def load_cifar10_data(img_rows, img_cols, nb_train_samples=1000,nb_test_samples=200,color=False):

	(x_train, y_train), (x_test, y_test) = cifar10.load_data()

	if K.image_data_format() == 'channels_first':
		x_train = x_train.reshape(x_train.shape[0], 3, img_rows, img_cols).astype('float32') / 255
		x_test = x_test.reshape(x_test.shape[0], 3, img_rows, img_cols).astype('float32') / 255
		input_shape = (1, img_rows, img_cols)
	else:
		x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 3).astype('float32') / 255
		x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 3).astype('float32') / 255
		input_shape = (img_rows, img_cols, 1)

	# convert to grayscale
	if color==False:
		x_train = np.expand_dims(np.dot(x_train[...,:3],[0.2989,0.5870,0.1140]),axis=x_train.shape[-1])
		x_test = np.expand_dims(np.dot(x_test[...,:3],[0.2989,0.5870,0.1140]),axis=x_test.shape[-1])
	
	# Transform targets to keras compatible format
	y_train = keras.utils.to_categorical(y_train, 10)
	y_test = keras.utils.to_categorical(y_test, 10)

	print("x_train shape:%s\ny_train shape:%s\nx_test shape:%s\ny_test shape:%s" % (str(x_train.shape),str(y_train.shape),str(x_test.shape),str(y_test.shape)))
	return x_train, y_train, x_test, y_test

def load_cifar100_data(img_rows, img_cols, nb_train_samples=1000,nb_test_samples=200,color=False):

	(x_train, y_train), (x_test, y_test) = cifar100.load_data()

	# Resize trainging images
	if K.image_dim_ordering() == 'th':
		x_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in x_train[:nb_train_samples,:,:,:]])
		x_test = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in x_test[:nb_test_samples,:,:,:]])
	else:
		x_train = np.array([cv2.resize(np.divide(img,255), (img_rows,img_cols)) for img in x_train[:nb_train_samples,:,:,:]])
		x_test = np.array([cv2.resize(np.divide(img,255), (img_rows,img_cols)) for img in x_test[:nb_test_samples,:,:,:]])

	# convert to grayscale
	if color==False:
		x_train = np.expand_dims(np.dot(x_train[...,:3],[0.2989,0.5870,0.1140]),axis=x_train.shape[-1])
		x_test = np.expand_dims(np.dot(x_test[...,:3],[0.2989,0.5870,0.1140]),axis=x_test.shape[-1])

	# Transform targets to keras compatible format
	y_train = np_utils.to_categorical(y_train[:nb_train_samples], 100)
	y_test = np_utils.to_categorical(y_test[:nb_test_samples], 100)

	print("x_train shape:%s\ny_train shape:%s\nx_test shape:%s\ny_test shape:%s" % (str(x_train.shape),str(y_train.shape),str(x_test.shape),str(y_test.shape)))
	return x_train, y_train, x_test, y_test


def plot_history(histories,nb_epoch, key='categorical_crossentropy'):
	plt.figure(figsize=(16,10))
	cmap = plt.get_cmap('jet_r')
	i = 1
	for name, history in histories:
		color = cmap(float(i)/len(histories))
		i = i+1
		plt.plot(history.history['acc'],linestyle='-',c=color,label=str(name+' acc'))
		plt.plot(history.history['val_acc'],linestyle='--',c=color,label=str(name+' val_acc'))

	plt.xlabel('Epochs')
	plt.ylabel(key.replace('_',' ').title())
	plt.legend()
	plt.xlim([0,max(history.epoch)])
	plt.show()

# FROM https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
	"""
	Freezes the state of a session into a pruned computation graph.

	Creates a new computation graph where variable nodes are replaced by
	constants taking their current value in the session. The new graph will be
	pruned so subgraphs that are not necessary to compute the requested
	outputs are removed.
	@param session The TensorFlow session to be frozen.
	@param keep_var_names A list of variable names that should not be frozen,
						  or None to freeze all the variables in the graph.
	@param output_names Names of the relevant graph outputs.
	@param clear_devices Remove the device directives from the graph for better portability.
	@return The frozen graph definition.
	"""
	graph = session.graph
	with graph.as_default():
		freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
		output_names = output_names or []
		output_names += [v.op.name for v in tf.global_variables()]
		input_graph_def = graph.as_graph_def()
		if clear_devices:
			for node in input_graph_def.node:
				node.device = ""
		frozen_graph = tf.graph_util.convert_variables_to_constants(
			session, input_graph_def, output_names, freeze_var_names)
		return frozen_graph