
import cv2
import numpy as np

from keras.datasets import cifar10
from keras import backend as K
from keras.utils import np_utils

num_classes = 10

def load_cifar10_data(img_rows, img_cols, nb_train_samples=1000,nb_valid_samples=200):

    # Load cifar10 training and validation sets
    (X_train, Y_train), (X_valid, Y_valid) = cifar10.load_data()

    # Resize trainging images
    if K.image_dim_ordering() == 'th':
        X_train = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(img.transpose(1,2,0), (img_rows,img_cols)).transpose(2,0,1) for img in X_valid[:nb_valid_samples,:,:,:]])
    else:
        X_train = np.array([cv2.resize(np.divide(np.dot(img[...,:3],[0.2989, 0.5870, 0.1140]),255), (img_rows,img_cols)) for img in X_train[:nb_train_samples,:,:,:]])
        X_valid = np.array([cv2.resize(np.divide(np.dot(img[...,:3],[0.2989, 0.5870, 0.1140]),255), (img_rows,img_cols)) for img in X_valid[:nb_valid_samples,:,:,:]])

    # Transform targets to keras compatible format
    Y_train = np_utils.to_categorical(Y_train[:nb_train_samples], num_classes)
    Y_valid = np_utils.to_categorical(Y_valid[:nb_valid_samples], num_classes)

    return X_train, Y_train, X_valid, Y_valid

if __name__ == "__main__":
    from matplotlib import pyplot as plt
    img_rows,img_cols = 32, 32
    X_train, Y_train, X_valid, Y_valid = load_cifar10_data(img_rows, img_cols)

    fig = plt.figure(figsize=(32,32))
    columns=3
    rows=4

    ax = []
    for i in range(columns*rows):
        img = X_train[i].squeeze()
        label = Y_train[i]
        ax.append(fig.add_subplot(rows,columns,i+1))
        ax[-1].set_title(str(label))
        plt.imshow(img)
    plt.show()