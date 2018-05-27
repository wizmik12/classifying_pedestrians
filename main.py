import numpy as np #supporting multi-dimensional arrays and matrices
import os #read or write a file

import cv2
from sklearn.metrics import confusion_matrix

from utils import (label_img, label_return, read_data, hog_descriptor, SVM_linear)



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

#Sacamos las caracteristicas HOG
hog_train_data = hog_descriptor(train_data.copy())
hog_test_data = hog_descriptor(test_data.copy())


#Clasificamos con HOG
clf = SVM_linear()

svm = cv2.ml.SVM_create()
svm.setType(cv2.ml.SVM_C_SVC)
svm.setKernel(cv2.ml.SVM_LINEAR)
# svm.setDegree(0.0)
# svm.setGamma(0.0)
# svm.setCoef0(0.0)
# svm.setC(0)
# svm.setNu(0.0)
# svm.setP(0.0)
# svm.setClassWeights(None)
svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))




hog_train_data = np.array(hog_train_data, dtype = np.float32)
train_labels = np.array(train_labels, dtype = np.int32)
test_labels = np.array(test_labels, dtype = np.int32)


svm.train(hog_train_data, cv2.ml.ROW_SAMPLE, train_labels)

hog_test_data = np.array(hog_test_data, dtype = np.float32)
predicted = svm.predict(hog_test_data)[1]

confusion_matrix(test_labels, predicted)