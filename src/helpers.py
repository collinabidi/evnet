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


def plot_history(histories,nb_epoch, key='categorical_crossentropy'):
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

# FROM https://stackoverflow.com/questions/45466020/how-to-export-keras-h5-to-tensorflow-pb
def freeze_session(session, keep_var_names=None, output_names=None, clear_devices=True):
    """
    Freezes the state of a session into a pruned computation graph.

    Creates a new computation graph where variable nodes are replaced by
    constants taking their current value in the session. The new graph will be
    pruned so subgraphs that are not necessary to compute the requested
    outputs are removed.
    @param session The TensorFlow session to be frozen.
    @param keep_var_names A list of variable names that should not be frozen,
                          or None to freeze all the variables in the graph.
    @param output_names Names of the relevant graph outputs.
    @param clear_devices Remove the device directives from the graph for better portability.
    @return The frozen graph definition.
    """
    graph = session.graph
    with graph.as_default():
        freeze_var_names = list(set(v.op.name for v in tf.global_variables()).difference(keep_var_names or []))
        output_names = output_names or []
        output_names += [v.op.name for v in tf.global_variables()]
        input_graph_def = graph.as_graph_def()
        if clear_devices:
            for node in input_graph_def.node:
                node.device = ""
        frozen_graph = tf.graph_util.convert_variables_to_constants(
            session, input_graph_def, output_names, freeze_var_names)
        return frozen_graph