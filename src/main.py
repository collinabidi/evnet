import numpy as np
import dask.array as da
import tensorflow as tf
import keras,random,cv2,os,sys,getopt
import math
import matplotlib

from keras import backend as K
from keras.utils import np_utils
from scipy import stats
from matplotlib import pyplot as plt
from matplotlib.collections import LineCollection
from matplotlib import colors as mcolors
from population import Population
from helpers import plot_history,load_cifar10_data,load_cifar100_data,load_mnist_data
from sklearn.model_selection import train_test_split

# get current working directory 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
cwd = os.getcwd()

def main(argv):
	img_rows, img_cols = 28, 28
	channel = 1
	num_classes = 10
	dataset = 'mnist'
	batch_size = 1028
	nb_epoch = 10
	train_num = 10000
	test_num = 3000
	pop_size = 15
	generations = 4
	
	try:
		opts, args = getopt.getopt(argv,"hi:v:",["dataset","color"])
	except getopt.GetoptError:
		print ('USAGE: main.py -d <dataset> -c <color>')
	for opt, args in opts:
		print("opt:%s args:%s" % (opt,args))
		if opts=='-h':
			print ('USAGE: main.py -d <dataset [mnist,cifar10,cifar100]> -c <color [rgb,grey]>')
		elif opt in ("-d", "--dataset"):
			print('got d')
			if opt is "mnist":
				dataset='mnist'
				print('using mnist')
			elif opt is "cifar10":
				print('using cifar10')
				dataset='cifar10'
				img_rows, img_cols = 32, 32
				num_classes = 10
			elif opt is 'cifar100':
				print('using cifar100')
				dataset='cifar100'
				img_rows, img_cols = 32, 32
				num_classes = 100
		elif opt in ("-c","--color"):
			if opt is "rgb":
				print('using RGB')
				channel = 3
			elif opt is "grey" or opt is "gray":
				print('using GRAYSCALE')
				channel = 1

	if dataset=='mnist':
		x_train, y_train, x_test, y_test = load_mnist_data(img_rows, img_cols, nb_train_samples=train_num,nb_test_samples=test_num)
	elif dataset=='cifar10':
		x_train, y_train, x_test, y_test = load_cifar10_data(img_rows, img_cols, nb_train_samples=train_num,nb_test_samples=test_num)
	elif dataset=='cifar100':
		x_train, y_train, x_test, y_test = load_cifar100_data(img_rows, img_cols, nb_train_samples=train_num,nb_test_samples=test_num)

	x_test, x_valid, y_test, y_valid = train_test_split(x_test, y_test, test_size=0.2)

	# lenet model for testing
	conv1 = {'name':'conv1','type':'Convolution2D','border_mode':'same','nb_filter':6,'nb_row':3,'nb_col':3,'activation':'relu','input_shape':(img_rows,img_cols,channel)}
	max1 = {'name':'max1','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
	conv2 = {'name':'conv2','type':'Convolution2D','border_mode':'same','nb_filter':16,'nb_row':3,'nb_col':3,'activation':'relu'}
	max2 = {'name':'max2','type':'MaxPooling2D','pool_size':(2,2),'strides':(2,2)}
	flatten1 = {'name':'flatten1','type':'Flatten'}
	dense1 = {'name':'dense2','type':'Dense','output_dim':120,'activation':'relu'}
	dense2 = {'name':'dense2','type':'Dense','output_dim':84,'activation':'relu'}
	dense3 = {'name':'dense2','type':'Dense','output_dim':num_classes,'activation':'softmax'}
	p = [conv1,max1,conv2,max2,flatten1,dense1,dense2,dense3]

	pop = Population(p,size=pop_size,k_best=3)

	for i in range(0,generations):
		if i > 0:
			pop.evolve()
		big_poppa = pop.train_evaluate_population(x_train,y_train,x_valid,y_valid,batch_size,nb_epoch,x_test,y_test)



	# plot all generations 
	# help from 
	# https://stackoverflow.com/questions/14777066/matplotlib-discrete-colorbar
	# https://stackoverflow.com/questions/11244514/modify-tick-label-text
	
	# concatenate separate histories
	lines = []
	name_list = []
	z = [0] * (generations*pop_size)
	for gen in range(0,generations):
		histories = pop.generation_histories[gen]
		i = 0
		for name, history in histories:
			if name not in name_list:
				lines.append(history.history['val_acc'])
				z[i+gen*pop_size] = gen
				i = i+1
				name_list.append(name)
	new_z = [0,1,2,3,4]
	# fill in values for early stopping!
	for i in range(0,len(lines)):
		while len(lines[i])<10:
			lines[i].append(lines[i][-1])
	
	fig, ax = plt.subplots(1,1,figsize=(20,10),dpi=120)

	# define colormap, extract colors from map
	cmap = plt.cm.BuPu
	cmaplist = [cmap(i) for i in range(cmap.N)]
	cmap = matplotlib.colors.LinearSegmentedColormap.from_list('Custom cmap',cmaplist,cmap.N)

	# define bins and normalize
	bounds = np.linspace(0,generations,generations+1)
	norm = matplotlib.colors.BoundaryNorm(bounds,cmap.N)

	# create plot
	x = np.arange(1,nb_epoch+1)
	coord = [np.column_stack((x,y)) for y in lines]
	line_c = LineCollection(coord,linestyle='solid',cmap=cmap,norm=norm)
	line_c.set_array(np.array(new_z))
	lineplot = ax.add_collection(line_c)

	# do colorbar stuff
	cb = fig.colorbar(line_c,cmap=cmap,norm=norm,ticks=bounds,boundaries=bounds,format='%1i')
	cb.set_label("Generation Number")
	ax.set_title("Evolution Summary")
	ax.set_xlabel("Epochs")
	ax.xaxis.set_ticks(np.arange(1,nb_epoch+1))
	ax.xaxis.set_major_formatter(ticker.FormatStrFormatter('%1i'))
	ax.set_ylabel("Accuracy")
	ax.set_yscale('logit')

	plt.show()


	print ("\n\n======== FINAL WINNER : " + big_poppa.name + " ========")


	# CONVERT AND SAVE TO TFLITE FILE
	print("\nConverting to TfLite File")
	keras_file = str(big_poppa.name)+".h5"
	converter = tf.contrib.lite.TFLiteConverter.from_keras_model_file("models/"+keras_file)
	tflite_model = converter.convert()
	open("models/winners/"+str(big_poppa.name)+".tflite", "wb").write(tflite_model)



if __name__ == "__main__":
	main(sys.argv[1:])