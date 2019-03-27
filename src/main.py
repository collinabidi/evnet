# based on https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html
import autokeras as ak
from load_cifar_10 import load_cifar10_data
from autokeras import ImageClassifier
from sklearn import preprocessing
import numpy as np

def load_images():
	img_rows, img_cols = 32,32
	X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)
	Y_train,Y_valid = np.array([np.argmax(i) for i in Y_train]),np.array([np.argmax(i) for i in Y_valid])

	print("X_train shape: %s\nY_train shape: %s\nX_valid shape: %s\nY_valid shape: %s" % (str(X_train.shape),str(Y_train.shape),str(X_valid.shape),str(Y_valid.shape)))
	return X_train, Y_train, X_valid, Y_valid


def run():
	x_train, y_train, x_test, y_test = load_images()
	x_train = x_train.reshape(x_train.shape+(1,))
	x_test = x_test.reshape(x_test.shape+(1,))
	print(y_train[10:])
	# After loading train and evaluate classifier.
	clf = ImageClassifier(verbose=True, augment=False)
	clf.fit(x_train, y_train, time_limit=12 * 60 * 60)
	clf.final_fit(x_train, y_train, x_test, y_test, retrain=True)
	y = clf.evaluate(x_test, y_test)
	print(y * 100)


if __name__ == '__main__':
	run()
