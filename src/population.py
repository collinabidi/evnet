#! python3
import random
import copy
import math

from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.optimizers import SGD
from sklearn.metrics import log_loss

from load_cifar_10 import load_cifar10_data

# population class defines the individual/gene and linked list that make up a representation of
# a population of individuals that are made up of layers
__all__ = ['Individual','Population']

class Layer:
	"""
	This class defines a general purpose Layer type that has unique parameters
	"""
	activation_list = ['tanh','softmax','relu','sigmoid']

	def __init__(self,gene):
		temp = []
		if type(gene) is Layer:
			temp = gene.as_list()
		else:
			temp = gene

		self.__dict__.update((k,v) for k,v in temp.items())
		print("\tinitializing layer " + self.__dict__['name'])
	
	def print_layer(self):
		print(self.__dict__)

	def as_list(self):
		return self.__dict__
	
	def mutate_layer(self):
		if 'activation' in self.__dict__:
			print('mutated activation')
			self.activation = random.choice(self.activation_list)
		#if 'strides' in self.__dict__:
		#	print('\tmutated stride')
		#	rand_factor = random.choice([0.5,1.0,2.0])
		#	for i in range(len(self.strides)):
		#		self.strides[i] = int(math.floor(self.strides[i] * rand_factor))

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
		print("initializing " + name)
		self.name = name
		self.layers = []
		for layer in gene:
			self.append_layer(layer)
		self.fitness = fitness
	
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

	# todo returns a new individual that's based on the original one but mutated
	def mutate(self, name="default name", prob=0.1):
		new_individual = copy.deepcopy(self.layers)
		for layer in new_individual:
			r = random.uniform(0.0,1.0)
			if r < prob:
				#print('\tmutating %s from %s' % (layer.activation,layer.name))
				layer.mutate_layer()
		return Individual(new_individual,name)

	# todo
	def crossover(self, mate):
		return
	
	# todo
	def evaluate_fitness(self, train_data):
		return

	def build_model(self):
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
		sgd = SGD(lr=1e-3, decay=1e-6, momentum=0.9, nesterov=True)
		model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

		return model



class Population:
	"""
	This class defines an unsorted population of Individuals for use in genetic algorithms
	"""
	population = []

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
		for individual in self.population:
			model = individual.build_model()
			model.fit(X_train, Y_train, batch_size=batch_size, nb_epoch=nb_epoch, shuffle=True, verbose=1, validation_data=(X_valid, Y_valid),)
			predictions_valid = model.predict(X_valid, batch_size=batch_size, verbose=1)
			score = log_loss(Y_valid, predictions_valid)
			print("Individual " + str(individual.name) + " fitness score: " + str(score))




if __name__ == '__main__':

	img_rows, img_cols = 224, 224 # Resolution of inputs
	channel = 3
	num_classes = 10

	# lenet model for testing
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
	activation4 = {'name':'activation4','type':'Activation','activation':'softmax'}


	p = [conv1,activation1,max1,conv2,activation2,max2, flatten1, dense1, activation3, dense2, activation4]

	pop = Population(p)

	#pop.print_population()


	# Example to fine-tune on 3000 samples from Cifar10
	batch_size = 16 
	nb_epoch = 10

	X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

	pop.train_evaluate_population(X_train,Y_train,batch_size,nb_epoch,X_valid,Y_valid)

