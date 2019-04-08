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

# from https://stackoverflow.com/questions/43178668/record-the-computation-time-for-each-epoch-in-keras-during-model-fit
class TimeHistory(keras.callbacks.Callback):
	def on_train_begin(self,logs={}):
		self.times = []
	def on_epoch_begin(self,batch,logs={}):
		self.epoch_time_start = time.time()
	def on_epoch_end(self,batch,logs={}):
		self.times.append(time.time()-self.epoch_time_start)


def augment_data(X,y,batch_size,augment_size):
	datagen = ImageDataGenerator(zca_whitening=True)
	datagen.fit(X)
	original_length = np.size(X,axis=0)
	batches = 0
	for X_batch, y_batch in datagen.flow(X, y, batch_size=original_length):
		X = np.concatenate((X,X_batch),axis=0)
		y = np.concatenate((y,y_batch),axis=0)
		print(y.shape)
		batches = batches + 1
		if batches >= augment_size:
			break
	return (X,y)

def plot_history(histories,nb_epoch, key='binary_crossentropy'):
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


def image_generator(X,y,bs,mode="train",aug=None):
	while True:
		
		if aug is not None:
			(X,y) = next(aug.flow(X,y,batch_size=bs))

		yield (X,y)