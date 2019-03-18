#! python3

# population class defines the individual/gene and linked list that make up a representation of
# a population of individuals that are made up of layers
__all__ = ['Individual','Population']

class Individual:
	"""
	This class defines a chromosome for the genetic algorithm. Each individual has
	a unique name, and a linked list of Layers that will be used when building the
	keras model from the Individual
	"""
	def __init__(self,gene=None):
		# initialize linked list using the input list of dictionaries
		for l in gene:
			layer = Layer()


class Layer:
	"""
	This class defines a general purpose Layer type that has unique name and size
	parameters.
	"""
	def __init__(self,name="Unnamed",layer_type="Unspecified Layer Type",size=[-1,-1,-1,-1],activation="No Activation Yet"):
		self.name = name
		self.layer_type = layer_type
		self.size = size
		self.activation = activation
		self.next_layer = None

	def set_layer_type(self,layer_type="New Layer Type"):
		self.layer_type = layer_type

	def set_name(self,name="New Name"):
		self.name = name

	def set_size(self,size="New Size"):
		self.size = size

	def set_activation(self,activation="New Activation"):
		self.activation = activation

	def get_next_layer(self):
		return self.next_layer