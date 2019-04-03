#! python3
# helper functions
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
from keras.backend.tensorflow_backend import set_session,clear_session,get_session
import tensorflow as tf
import numpy as np
import gc


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

def reset_keras():
	sess = get_session()
	clear_session()
	sess.close()
	sess = get_session()

	try:
		del classifier
	except:
		pass

	print(gc.collect())

	config = tf.ConfigProto()
	config.gpu_options.per_process_gpu_memory_fraction = 0.90
	config.gpu_options.visible_device_list="0"
	set_session(tf.Session(config=config))