import cv2
import numpy as np
import random

from keras.datasets import cifar10,cifar100
from keras import backend as K
from keras.utils import np_utils
from scipy import stats

def load_cifar10_data(img_rows, img_cols, nb_train_samples=5000,nb_test_samples=500):

    # Load cifar10 training and test sets
    (X_train, Y_train), (X_test, Y_test) = cifar10.load_data()

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_test = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_test[:nb_test_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_test = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_test[:nb_test_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], 10)
    Y_test = np_utils.to_categorical(Y_test[:nb_test_samples], 10)
    
    return X_train, Y_train, X_test, Y_test

def load_cifar100_data(img_rows, img_cols, nb_train_samples=15000,nb_test_samples=1000):

    # Load cifar100 training and testing sets
    (X_train, Y_train), (X_test, Y_test) = cifar100.load_data()

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_test = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_test[:nb_test_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_test = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_test[:nb_test_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], 100)
    Y_test = np_utils.to_categorical(Y_test[:nb_test_samples], 100)

    return X_train, Y_train, X_test, Y_test

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    from helpers import augment_data

    img_cols,img_rows = 32,32
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols, nb_train_samples=100,nb_test_samples=100)

    #X_train, Y_train = augment_data(X_train,Y_train,50,2)
    #Y_train,Y_valid = np.array([np.argmax(i) for i in Y_train]),np.array([np.argmax(i) for i in Y_valid])

    print("X_train shape: " + str(X_train.shape))
    print("Y_train shape: " + str(Y_train.shape))

    fig = plt.figure(figsize=(32,32))
    columns=3
    rows=4
    f,axarr = plt.subplots(rows,columns)
    num = 0
    for i in range(0,rows):
        for j in range(0,columns):
            img = X_train[num]
            label = Y_train[num]

            axarr[i,j].set_title(label)
            axarr[i,j].imshow(img)

            num = num + 1
            
    plt.show()