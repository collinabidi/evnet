#! python3
# helper functions
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot as plt
import numpy as np

def augment_data(X,y,batch_size,augment_size):
	datagen = ImageDataGenerator(zca_whitening=True)
	datagen.fit(X)
	original_length = np.size(X,axis=0)
	batches = 0
	for X_batch, y_batch in datagen.flow(X, y, batch_size=batch_size):
		X = np.concatenate((X,X_batch),axis=0)
		y = np.concatenate((y,y_batch),axis=0)
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