#! python3
import random
import copy
import math

import numpy as np

from keras.models import Sequential
from keras.utils import plot_model
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD, Adam
from sklearn.metrics import log_loss
from scipy.interpolate import make_interp_spline, BSpline

from load_cifar_10 import load_cifar10_data

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

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

	# converts the layer to a string vector representation
	# FORMAT: type,filter_height,filter_width,stride_height,stride_width,num_filters!
	def layer_to_vec(self):
		return str(self.type + ',' + self.filter_height+','+self.filter_width+','+self.stride_height+','+self.stride_width+','+self.num_filters+'!')
	
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

	def train_evaluate_population(self,X_train,Y_train,batch_size,nb_epoch,X_valid,Y_valid):
		print("TRAINING GRANNY")
		model = self.model.build_model(learn_rate=0.001)
		history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1,validation_split=0.2)
		predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)

		for individual in self.population:
			model = individual.build_model(learn_rate=0.001)
			history = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1,validation_split=0.2)
			predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
			score = log_loss(Y_valid, predictions_valid)
			print("Individual " + str(individual.name) + " fitness score: " + str(score))
			individual.set_fitness(score)
			self.histories.append(history)

		# sort the results
		sorted_individuals = sorted(self.population, key=lambda x: x.fitness)
		sorted_results = {i.name:i.fitness for i in sorted_individuals}
		print(sorted_results)

		# plot top k results
		k = 3
		plot_history([(name,history) for name, history in zip([i.name for i in sorted_individuals],self.histories)],nb_epoch)


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
	channel = 1 # greyscale
	num_classes = 10

	# lenet model for testing
	conv1 = {'name':'conv1','type':'Convolution2D','border_mode':'same','nb_filter':20,'nb_row':5,'nb_col':5,'input_shape':(channel,img_rows,img_cols)}
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

	# create population
	p = [conv1,activation1,max1,conv2,activation2,max2, flatten1, dense1, activation3, dense2, activation4]
	pop = Population(p,size=2)

	# Example to fine-tune on samples from Cifar10
	batch_size = 64 
	nb_epoch = 100
	X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols, nb_train_samples=10000,nb_valid_samples=2000)
	X_train,X_valid = X_train[:,None,:,:], X_valid[:,None,:,:]
	
	# add noise to training data
	std = np.std(X_train)
	mean = np.mean(X_train)
	noisy = X_train + np.random.normal(mean,std*0.3,X_train.shape)
	X_train_noisy = np.clip(noisy,0,255)

	# run train and evalute
	pop.train_evaluate_population(X_train_noisy,Y_train,batch_size,nb_epoch,X_valid,Y_valid)