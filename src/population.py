#! python3
import random
import copy
import math
import numpy as np
import os
import time
import tensorflow as tf

from keras.models import Sequential
from keras.utils import plot_model
from keras.layers.convolutional import Conv2D,MaxPooling2D
from keras.layers.core import Activation,Flatten,Dense
from keras.layers import GaussianNoise
from keras.optimizers import SGD, Adam
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.models import model_from_json
from sklearn.metrics import log_loss
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from keras import initializers


# get current working directory and set random seed
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cwd = os.getcwd()
seed = 7
np.random.seed(seed)

# set gpu memory usage options
config = tf.ConfigProto()
config.gpu_options.allow_growth=True
config.gpu_options.per_process_gpu_memory_fraction=0.5
K.tensorflow_backend.set_session(tf.Session(config=config))


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
		if 'activation' in self.__dict__ and self.name is not 'output' and self.__dict__['activation'] is not 'softmax':
			self.activation = random.choice(['tanh','relu','sigmoid'])
			print("\tmutated activation of %s to %s" % (str(self.name),str(self.activation)))


	def mutate_kernel(self):
		if 'nb_row' in self.__dict__ and self.name is not 'output':
			rand_factor = random.choice([0.5,1.0,2.0])
			self.nb_row = int(math.floor(self.nb_row* rand_factor))
			print("\tmutated nb_row of %s to %s" % (str(self.name),str(self.nb_row)))

class Individual:
	"""
	This class defines a chromosome for the genetic algorithm. Each individual has
	a unique name, and a list of Layers that will be used when building the
	keras model from the Individual
	"""

	layers = []
	train_time = 0

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
	def mutate(self, name="default name", prob=0.2):
		new_individual = copy.deepcopy(self.layers)
		for layer in new_individual:
			r = random.uniform(0.0,1.0)
			if r < prob and layer.name is not 'output':
				layer.mutate_activation()
			r = random.uniform(0.0,1.0)
			if r < prob and layer.name is not 'output':
				print("KERNEL MUTATION OF %s" % (str(name)))
				layer.mutate_kernel()
		return Individual(new_individual,name)

	# returns a copy
	def new_copy(self, name="default name",prob=0.1):
		new_individual = copy.deepcopy(self.layers)
		return Individual(new_individual,name)

	def build_model(self,learn_rate=0.001):
		model = Sequential()
		for layer in self.layers:
			print(layer.__dict__)
			if layer.type is 'Convolution2D':
				if 'input_shape' in layer.__dict__:
					model.add(Conv2D(layer.nb_filter,layer.nb_row,activation=layer.activation,padding=layer.border_mode,input_shape=layer.input_shape,kernel_initializer="random_normal"))
					model.add(GaussianNoise(0.1))
				else:
					model.add(Conv2D(layer.nb_filter,layer.nb_row,activation=layer.activation,padding=layer.border_mode,kernel_initializer="random_normal"))
			elif layer.type is 'MaxPooling2D':
				model.add(MaxPooling2D(pool_size=layer.pool_size,strides=layer.strides,data_format="channels_first"))
			elif layer.type is 'Dense':
				model.add(Dense(layer.output_dim,activation=layer.activation,kernel_initializer="random_normal"))
			elif layer.type is 'Flatten':
				model.add(Flatten())
			elif layer.type is 'Dropout':
				model.add(Dropout(layer.p))

		sgd = SGD(lr=learn_rate, decay=1e-6, momentum=0.9, nesterov=True)
		#adam = Adam(lr=learn_rate,beta_1=0.9,beta_2=0.999,epsilon=1e-08)
		model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

		return model


class Population:

	population = []
	histories = []
	gen_id = 0
	generation_histories = {}

	def __init__(self,model,size=10,crossover=0.8,k_best=10,mutation=0.2):
		print("initializing population")
		self.model = Individual(model,"Grandparent")
		self.size = size
		self.crossover = crossover
		self.k_best = k_best
		self.mutation = mutation

		for i in range(size):
			self.population.append(self.model.mutate(prob=mutation,name=("# "+str(self.gen_id) + "_" +str(i))))
		
		print("done initializing population")

	def print_population(self):
		for individual in self.population:
			individual.print_individual()


	def train_evaluate_population(self,X,Y,X_valid,Y_valid,batch_size,nb_epoch,X_test,Y_test):
		print("\n**************** TRAINING ****************\n")
		Y_test = np.argmax(np.swapaxes(Y_test,0,1),axis=0)
		self.population.append(self.model)
		callbacks = [EarlyStopping(monitor='val_acc',min_delta=0.005,patience=3,mode='max',verbose=1)]
		for individual in self.population:
			K.clear_session() # keep backend clean

			print("NAME: " + str(individual.name))
			# build, fit, score model
			model = individual.build_model(learn_rate=0.005)
			
			start_time = time.time()
			history = model.fit(X, Y, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1,validation_data=(X_valid,Y_valid),callbacks=callbacks)
			end_time = time.time()
			individual.start_time, individual.end_time, individual.train_time = start_time, end_time, end_time-start_time 	

			predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)
			predictions_valid = np.argmax(np.swapaxes(predictions_valid,0,1),axis=0)
			acc = np.sum(predictions_valid == Y_test) / len(Y_test) * 100
			individual.set_fitness(acc)
			print("Final Accuracy: " + str(acc) + "%")

			self.histories.append(history)
				# save model
				#model_json = model.to_json()
				#path_name = cwd+("/models/model_"+str(individual.name)+" " +str(individual.fitness))
				#with open(path_name+".json","w") as json_file:
				#	json_file.write(model_json)
				# save weights
				#model.save_weights(path_name+".h5")
		
		# sort the results
		self.population.sort(key=lambda x: x.fitness)

		# plot top k results and save to generation histories
		h = [(name,history) for name, history in zip([i.name for i in self.population[:self.k_best]],self.histories[:self.k_best])]
		self.generation_histories[self.gen_id] = h

		print("\n******************************************\n")


	# turns population into the next generation
	def evolve(self):
		print("\n**************** EVOLVING ****************\n")
		new_pop = []
		self.gen_id = self.gen_id + 1
		self.population.sort(key=lambda x: x.fitness)
		# add winners
		for i in self.population[:self.k_best]:
			new_pop.append(i)
			print(" WINNER ===========> " + i.name + " :" + str(i.fitness))
		# generate children based on winners until we run out of space in the population
		for i in range(self.k_best,self.size):
			parent = self.population[i % self.k_best]
			new_pop.append(parent.mutate(prob=self.mutation,name=("# " + str(self.gen_id) + "_" + str(i))))

		self.population = new_pop
		print("\n******************************************\n")
