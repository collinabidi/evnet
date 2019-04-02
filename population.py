#! python3
import random
import copy
import math
import numpy as np

from keras.models import Sequential
from keras.utils import plot_model
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json

from sklearn.metrics import log_loss
from scipy.misc import toimage
from scipy.interpolate import make_interp_spline, BSpline

from load_cifar_10 import load_cifar10_data, load_cifar100_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cwd = os.getcwd()

seed = 7
np.random.seed(seed)

# population class defines the individual/gene and linked list that make up a representation of
# a population of individuals that are made up of layers
__all__ = ['Individual','Population']

class Layer:
	"""
	This class defines a general purpose Layer type that has unique parameters
	"""
	filter_height, filter_width, stride_height, stride_width, num_filters = 0,0,0,0,0

	def __init__(self,gene):
		temp = []
		if type(gene) is Layer:
			temp = gene.as_list()
		else:
			temp = gene

		self.__dict__.update((k,v) for k,v in temp.items())
	
	def print_layer(self):
		print(self.__dict__)

	def as_list(self):
		return self.__dict__
	
	def mutate_activation(self):
		if 'activation' in self.__dict__ and self.name is not 'output':
			self.activation = random.choice(['tanh','relu','sigmoid'])

	def mutate_kernel(self):
		if 'nb_row' in self.__dict__ and self.name is not 'output':
			rand_factor = random.choice([0.5,1.0,2.0])
			self.nb_row = (int(math.floor(self.nb_row* rand_factor)),int(math.floor(self.nb_row * rand_factor)))

class Individual:
	"""
	This class defines a chromosome for the genetic algorithm. Each individual has
	a unique name, and a list of Layers that will be used when building the
	keras model from the Individual
	"""

	# encoded version of the layers in case we want to use it for stuff idk
	gene = ""
	layers = []

	def __init__(self,gene,name,fitness=-1):
		self.name = name
		self.layers = []
		for layer in gene:
			self.append_layer(layer)
		self.fitness = fitness
		self.num_layers = len(self.layers)

	
	def __iter__(self):
		return self

	def append_layer(self, l):
		layer = Layer(l)
		self.layers.append(layer)
	
	def print_individual(self):
		print("Individual " + self.name)
		for l in self.layers:
			l.print_layer()
		return

	def set_fitness(self,score):
		self.fitness = score

	# todo returns a new individual that's based on the original one but mutated
	def mutate(self, name="default name", prob=0.1):
		new_individual = copy.deepcopy(self.layers)
		for layer in new_individual:
			r = random.uniform(0.0,1.0)
			if r < prob and layer.name is not 'output':
				layer.mutate_activation()
			r = random.uniform(0.0,1.0)
			if r < prob and layer.name is not 'output':
				layer.mutate_kernel()
		return Individual(new_individual,name)

	# returns a copy
	def new_copy(self, name="default name", prob=0.1):
		new_individual = copy.deepcopy(self.layers)
		return Individual(new_individual,name)

	def build_model(self,learn_rate=0.001):
		model = Sequential()
		for layer in self.layers:
			print(layer.__dict__)
			if layer.type is 'Convolution2D' and 'input_shape' in layer.__dict__:
				model.add(Conv2D(layer.nb_filter,layer.nb_row,padding=layer.border_mode,input_shape=layer.input_shape))
			elif layer.type is 'Convolution2D' and 'input_shape' not in layer.__dict__:
				model.add(Conv2D(layer.nb_filter,layer.nb_row,padding=layer.border_mode))
			elif layer.type is 'MaxPooling2D':
				model.add(MaxPooling2D(pool_size=layer.pool_size,strides=layer.strides,data_format="channels_first"))
			elif layer.type is 'Activation':
				model.add(Activation(layer.activation))
			elif layer.type is 'Dense':
				model.add(Dense(layer.output_dim))
			elif layer.type is 'Flatten':
				model.add(Flatten())
			elif layer.type is 'Dropout':
				model.add(Dropout(layer.p))
			elif layer.type is 'ZeroPadding2D':
				model.add(ZeroPadding2D(strides=layer.stride))

		# Learning rate is changed to 0.001
		sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
		adam = Adam(lr=learn_rate,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
		model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

		return model


class Population:
	"""
	This class defines an unsorted population of Individuals for use in genetic algorithms
	"""
	population = []
	histories = []

	def __init__(self,model,size=10,crossover=0.8,elitism=0.1,mutation=0.5):
		print("initializing population")
		self.model = Individual(model,"Grandparent")
		self.size = size
		self.crossover = crossover
		self.elitism = elitism
		self.mutation = mutation

		for i in range(size):
			self.population.append(self.model.mutate(prob=mutation,name=("#"+str(i))))
		
		print("done initializing population")

	def print_population(self):
		for individual in self.population:
			individual.print_individual()

	def train_evaluate_population(self,X_train,Y_train,batch_size,nb_epoch,X_valid,Y_valid,augment_ratio=2):
		# AUGMENT DATA
		datagen = ImageDataGenerator(zca_whitening=True)
		datagen.fit(X_train)
		original_length = np.size(X_train,axis=0)
		batches = 0
		for X_batch, y_batch in datagen.flow(X_train, Y_train, batch_size=original_length):
			X_train = np.concatenate((X_train,X_batch),axis=0)
			Y_train = np.concatenate((Y_train,y_batch),axis=0)
			print(X_train.shape)
			batches = batches + 1
			if batches >= augment_ratio:
				break


		self.population.append(self.model)
		model = self.model.build_model(learn_rate=0.003)
		history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1,validation_split=0.2)
		predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
		self.histories.append(history)

		
		for individual in self.population:
			model = individual.build_model(learn_rate=0.003)
			history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1,validation_split=0.2)
			predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
			score = log_loss(Y_valid, predictions_valid)
			individual.set_fitness(score)
			self.histories.append(history)

			# save model
			#model_json = model.to_json()
			#path_name = "C:\Users\quaza\git\evnet-master\evnet-master\src\dart"
			#with open(path_name+".json","w") as json_file:
			#	json_file.write(model_json)
			# save weights
			#model.save_weights(path_name+".h5")
		
		# sort the results
		sorted_individuals = sorted(self.population, key=lambda x: x.fitness)
		sorted_results = {i.name:i.fitness for i in sorted_individuals}
		print(sorted_results)

		# plot top k results
		k = 5
		plot_history([(name,history) for name, history in zip([i.name for i in sorted_individuals[:k]],self.histories[:k])],nb_epoch)




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


if __name__ == '__main__':
	from matplotlib import pyplot as plt


	img_rows, img_cols = 32, 32 # Resolution of inputs
	channel = 3# rgb
	num_classes = 100

	# vgg16 model for testing
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
	p = [conv1,activation1,max1,conv2,activation2,max2, flatten1, dense1, activation3, dense2, activation4]
	pop = Population(p,size=30)

	# Example to fine-tune on samples from Cifar10
	batch_size = 64 
	nb_epoch = 35
	X_train, Y_train, X_valid, Y_valid = load_cifar100_data(img_rows, img_cols, nb_train_samples=300,nb_valid_samples=500)
	X_train,X_valid = X_train.astype('float32'), X_valid.astype('float32')
	# run train and evalute
	pop.train_evaluate_population(X_train,Y_train,batch_size,nb_epoch,X_valid,Y_valid,augment_ratio=4)