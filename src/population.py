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
from keras.optimizers import SGD, Adam, Adadelta
from sklearn import metrics
from keras import backend as K
from sklearn.metrics import log_loss
from keras.callbacks import EarlyStopping, ModelCheckpoint
from sklearn.model_selection import train_test_split
from keras import initializers
from helpers import freeze_session

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
			# make sure no zero kernels happen
			new_val = 0
			while new_val == 0:
				rand_factor = random.choice([0.5,1.0,2.0])
				new_val = int(math.floor(self.nb_row* rand_factor))
			self.nb_row = new_val
			print("\tmutated nb_row of %s to %s" % (str(self.name),str(self.nb_row)))

class Individual:
	"""
	This class defines a chromosome for the genetic algorithm. Each individual has
	a unique name, and a list of Layers that will be used when building the
	keras model from the Individual
	"""

	layers = []
	train_time = 0
	num_parameters = 0
	trained = False
	gen_id = 0
	def __init__(self,gene,name,fitness=-1):
		self.name = name
		self.layers = []
		for layer in gene:
			self.append_layer(layer)
		self.fitness = fitness
		self.num_layers = len(self.layers)
		self.trained = False

	
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
	def mutate(self, name="default name", prob=0.2, gen_id=0):
		self.gen_id = gen_id
		new_individual = copy.deepcopy(self.layers)
		for layer in new_individual:
			r = random.uniform(0.0,1.0)
			if r < prob and layer.name is not 'output':
				layer.mutate_activation()
			r = random.uniform(0.0,1.0)
			if r < prob and layer.name is not 'output' and 'nb_row' in layer.__dict__:
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
			if layer.type is 'Convolution2D':
				if 'input_shape' in layer.__dict__:
					model.add(Conv2D(filters=layer.nb_filter,kernel_size=(layer.nb_row,layer.nb_row),activation=layer.activation,padding=layer.border_mode,input_shape=layer.input_shape))
					#model.add(GaussianNoise(0.1))
				else:
					model.add(Conv2D(filters=layer.nb_filter,kernel_size=(layer.nb_row,layer.nb_row),activation=layer.activation,padding=layer.border_mode))
			elif layer.type is 'MaxPooling2D':
				model.add(MaxPooling2D(pool_size=layer.pool_size,strides=layer.strides,data_format="channels_last"))
			elif layer.type is 'Dense':
				model.add(Dense(units=layer.output_dim,activation=layer.activation))
			elif layer.type is 'Flatten':
				model.add(Flatten())
			elif layer.type is 'Dropout':
				model.add(Dropout(layer.p))

		#sgd = SGD(lr=0.1,decay=0.0, momentum=0.6)
		#adam = Adam(lr=0.002,beta_1=0.9,beta_2=0.999,epsilon=1e-08,decay=0.0)
		model.compile(optimizer=Adadelta(), loss='categorical_crossentropy', metrics=['accuracy'])

		return model


class Population:

	population = []
	histories = []
	gen_id = 0
	generation_histories = {}
	original_accuracy=0

	def __init__(self,model,size=10,crossover=0.8,k_best=10,mutation=0.3):
		print("initializing population")
		self.model = Individual(model,"Original")
		self.size = size
		self.crossover = crossover
		self.k_best = k_best
		self.mutation = mutation
		self.gen_id = 0

		for i in range(size):
			self.population.append(self.model.mutate(prob=mutation,gen_id = 0,name=("Gen"+str(self.gen_id) + "_Ind" +str(i))))
		
		print("done initializing population")

	def print_population(self):
		for individual in self.population:
			individual.print_individual()


	def train_evaluate_population(self,X,Y,X_valid,Y_valid,batch_size,nb_epoch,X_test,Y_test):
		print("\n**************** TRAINING ****************\n")
		Y_test = np.argmax(np.swapaxes(Y_test,0,1),axis=0)
		for individual in self.population:
			K.clear_session() # keep backend clean

			# BUILD MODEL
			if individual.trained == True:
				print("Already trained " + str(individual.name))
			else:
				model = individual.build_model(learn_rate=0.005)
				print("\n\nNAME: " + str(individual.name))
				print(model.summary())
				
				callbacks = [EarlyStopping(monitor='val_acc',min_delta=0.005,patience=3,mode='max',verbose=1),ModelCheckpoint("models/"+individual.name+".ckpt",save_weights_only=True,verbose=1)]

				# FIT MODEL and record times
				start_time = time.time()
				history = model.fit(X, Y, batch_size=batch_size, epochs=nb_epoch, shuffle=True, verbose=1,validation_data=(X_valid,Y_valid),callbacks=callbacks)
				end_time = time.time()
				individual.start_time, individual.end_time, individual.train_time = start_time, end_time, end_time-start_time 	

				# SCORE MODEL
				predictions_valid = model.predict(X_test, batch_size=batch_size, verbose=1)
				predictions_valid = np.argmax(np.swapaxes(predictions_valid,0,1),axis=0)
				acc = np.sum(predictions_valid == Y_test) / len(Y_test) * 100
				if individual.name=="Original":
					print("\n---------- ORIGINAL ACCURACY " + str(acc) + "% ----------")
					original_accuracy = acc
				else:
					print("\n---------- FINAL ACCURACY " + str(acc) + "% ----------")
				individual.set_fitness(acc)
				individual.num_parameters = model.count_params()
				# confused? me too
				#print("\n############ CONFUSION MATRIX ############")
				#matrix = metrics.confusion_matrix(Y_test, predictions_valid)
				#print(matrix)
				#print("\n##########################################")

				# save the history for graphing
				self.histories.append(history)

				# Save tf.keras model in HDF5 format.
				model.save_weights("models/"+str(individual.name)+".h5")
				# serialize model to JSON
				model_json = model.to_json()
				with open("models/"+str(individual.name)+".json", "w") as json_file:
					json_file.write(model_json)
				# make sure we don't train again
				individual.trained = True
		
		# sort the results
		self.population.sort(key=lambda x: x.fitness)
		self.population.reverse()

		# save results to generation histories
		h = [(name,history) for name, history in zip([i.name for i in self.population],self.histories)]
		self.generation_histories[self.gen_id] = h
		return self.population[0]
		print("\n******************************************\n")


	# turns population into the next generation
	def evolve(self):
		print("\n**************** EVOLVING ****************\n")
		new_pop = []
		old_pop = []
		self.gen_id = self.gen_id + 1

		self.population.sort(key=lambda x: x.fitness)
		self.population.reverse()
		# add winners
		for i in self.population[:self.k_best]:
			old_pop.append(i)
			print("WINNER ==> " + i.name + " :" + str(i.fitness))
		# generate children based on winners until we run out of space in the population
		for i in range(0,self.size):
			parent = old_pop[i % self.k_best]
			new_pop.append(parent.mutate(prob=0.2,gen_id=self.gen_id,name=("Gen"+str(self.gen_id) + "_Ind" +str(i))))
		
		self.population = new_pop
		return self.population[0]
		print("\n******************************************\n")
