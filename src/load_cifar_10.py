
import cv2
import numpy as np
import random

from keras.datasets import cifar10,cifar100
from keras import backend as K
from keras.utils import np_utils

from scipy import stats

num_classes = 10

def load_cifar10_data(img_rows, img_cols, nb_train_samples=1000,nb_valid_samples=200):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(np.divide(img,255), (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(np.divide(img,255), (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])
        """
        X_train = np.array([cv2.resize(np.divide(np.dot(img[...,:3],[0.2989, 0.5870, 0.1140]),255), (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(np.divide(np.dot(img[...,:3],[0.2989, 0.5870, 0.1140]),255), (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])
        """

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid

def load_cifar100_data(img_rows, img_cols, nb_train_samples=1000,nb_valid_samples=200):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar100.load_data()

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        #X_train = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        #X_valid = np.array([cv2.resize(img, (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])
        X_train = np.array([cv2.resize(np.divide(img,255), (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(np.divide(img,255), (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], 100)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], 100)

    return X_train, Y_train, X_valid, Y_valid

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    img_cols,img_rows = 32,32
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols, nb_train_samples=1000,nb_valid_samples=500)
    Y_train,Y_valid = np.array([np.argmax(i) for i in Y_train]),np.array([np.argmax(i) for i in Y_valid])
    print(Y_train[:10])
    fig = plt.figure(figsize=(32,32))
    columns=3
    rows=4

    ax = []
    rand_list = [random.randint(0,np.size(X_train,0))for _ in range(columns*rows)]
    for i in range(0,len(rand_list)):
        img = X_train[rand_list[i]]
        label = Y_train[rand_list[i]] 
        ax.append(fig.add_subplot(rows,columns,i+1))
        ax[-1].set_title(str(label))
        plt.imshow(img)
    plt.show()