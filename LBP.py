import numpy as np #supporting multi-dimensional arrays and matrices
import os #read or write a file

import cv2
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold


from utils import (read_data, hog_descriptor, LBP)



#Paths
data_dir = 'D://Master//ECI//Practica//Practica1//data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

#Clases
CATEGORIES = ['background', 'pedestrians']
NUM_CATEGORIES = len(CATEGORIES)

#Creamos los conjuntos de train y test
train_data, train_labels = read_data(train_dir, CATEGORIES)
test_data, test_labels = read_data(test_dir, CATEGORIES)

train_LBP = LBP(train_data)
test_LBP = LBP(test_data)