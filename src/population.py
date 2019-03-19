#! python3
import random
import copy

# population class defines the individual/gene and linked list that make up a representation of
# a population of individuals that are made up of layers
__all__ = ['Individual','Population']

class Layer:
	"""
	This class defines a general purpose Layer type that has unique parameters
	"""
	activation_list = ['relu','tanh','softmax','sigmoid']

	def __init__(self,gene):
		if type(gene) is not Layer:
			self.name = gene['name']
			self.activation = gene['activation']
			self.parents = gene['parents']
			self.children = gene['children']
		else:
			self.name = gene.name
			self.activation = gene.activation
			self.parents = gene.parents
			self.children = gene.children

	def get_children(self):
		return self.children
	
	def set_children(self,child):
		self.children = child
	
	def set_parents(self,parents):
		self.parents = parents
	
	def set_activation(self,activation):
		self.activation = activation
	
	def print_layer(self):
		print(self.__dict__)
	
	# mutate the layer based on input parameters
	def mutate_layer(self):
		self.activation = random.choice(self.activation_list)

class Individual:
	"""
	This class defines a chromosome for the genetic algorithm. Each individual has
	a unique name, and a linked list of Layers that will be used when building the
	keras model from the Individual
	"""
	edges = {}
	layers = []

	def __init__(self,gene,fitness=-1):
		# initialize linked list using the input list of dictionaries
		self.head = None
		self.print_individual()
		self.layers = []
		self.edges = {}
		for layer in gene:
			self.append_layer(layer)
		self.fitness = fitness
	
	def __iter__(self):
		return self

	def append_layer(self, l):
		layer = Layer(l)
		self.layers.append(layer)
		self.edges[layer.name] = layer.children			
	
	def print_individual(self):
		for l in self.layers:
			l.print_layer()

	# returns a new individual that's based on the original one but mutated
	def mutate(self, prob=0.3):
		new_individual = copy.deepcopy(self.layers)
		for layer in new_individual:
			r = random.uniform(0.0,1.0)
			if r < prob:
				#print('\tmutating %s from %s' % (layer.activation,layer.name))
				layer.mutate_layer()
		return Individual(new_individual)


class Population:
	"""
	This class defines an unsorted population of Individuals for use in genetic algorithms
	"""
	population = []

	def __init__(self,model,size=10,crossover=0.8,elitism=0.1,mutation=0.5):
		print("initializing population")
		self.model = Individual(model)
		self.size = size
		self.crossover = crossover
		self.elitism = elitism
		self.mutation = mutation

		for i in range(size):
			self.population.append(self.model.mutate(mutation))
		
		print("done initializing population")

	def print_population(self):
		for individual in self.population:
			individual.print_individual()




if __name__ == '__main__':
	p = []
	input_layer = {'name':'input_layer','activation':'relu','parents':None,'children':'conv1'}
	conv1 = {'name':'conv1','activation':'relu','parents':'input','children':'conv2'}
	conv2 = {'name':'conv2','activation':'relu','parents':'conv1','children':'max1'}
	max1 = {'name':'max1','activation':'softmax','parents':'conv2','children':'output'}
	output = {'name':'output_layer','activation':'softmax','parents':'max1','children':None}

	p = [input_layer,conv1,conv2,max1,output]
	pop = Population(p)
	pop.print_population()
