from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import numpy as np
from load_cifar_10 import load_cifar100_data
import os
cwd = os.getcwd()
if __name__ == "__main__":
	X_train, Y_train, X_valid, Y_valid = load_cifar100_data(32, 32, nb_train_samples=300,nb_valid_samples=5000)
	X_train,X_valid = X_train.astype('float32'), X_valid.astype('float32')

	# load json and create model
	json_file = open(cwd+"/models/model_16.134066192433238.json", 'r')
	loaded_model_json = json_file.read()
	json_file.close()
	loaded_model = model_from_json(loaded_model_json)
	# load weights into new model
	loaded_model.load_weights(cwd+"/models/model_16.134066192433238.h5")
	print("Loaded model from disk")
	x_false = np.random.rand(X_valid.shape[0],X_valid.shape[1],X_valid.shape[2],X_valid.shape[3])
	y_false = np.random.rand(Y_valid.shape[0],Y_valid.shape[1])
	print(X_valid.shape)
	print(Y_valid.shape)
	# evaluate loaded model on test data
	loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
	score = loaded_model.evaluate(X_valid, Y_valid, verbose=0)
	print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))