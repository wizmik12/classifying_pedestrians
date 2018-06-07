import numpy as np #supporting multi-dimensional arrays and matrices
import os #read or write a file

import cv2
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import KFold


from utils import (read_data, LBP)



#Paths
data_dir = 'D://Master//ECI//Practica//Practica_Caracteristicas//data'
train_dir = os.path.join(data_dir, 'train')
test_dir = os.path.join(data_dir, 'test')

#Clases
CATEGORIES = ['background', 'pedestrians']
NUM_CATEGORIES = len(CATEGORIES)

#Creamos los conjuntos de train y test
train_data, train_labels = read_data(train_dir, CATEGORIES)
test_data, test_labels = read_data(test_dir, CATEGORIES)

train_LBP = LBP(train_data.copy())
test_LBP = LBP(test_data.copy())

#Clasificamos con HOG
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



#Converting the data to array
train_LBP = np.array(train_LBP, dtype = np.float32)
train_LBP2 = np.expand_dims(train_LBP, axis=2)
train_labels = np.array(train_labels, dtype = np.int32)
test_labels = np.array(test_labels, dtype = np.int32)
test_LBP = np.array(test_LBP, dtype = np.float32)
test_LBP2 = np.expand_dims(test_LBP, axis=2)


#Training svm
svm.train(train_LBP, cv2.ml.ROW_SAMPLE, train_labels)

#Predicting
predicted = svm.predict(test_LBP)[1]

#Confusion matrix and precision on test
conf_mat = confusion_matrix(test_labels, predicted)
precision_test = accuracy_score(test_labels, predicted)

#Joining train and test
dataset = np.concatenate((train_LBP, test_LBP))
labels = np.concatenate((train_labels, test_labels))


#Cross Validation
kf = KFold(n_splits=10)
precision_list = []
for train, test in kf.split(dataset):
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
    svm.train(dataset[train], cv2.ml.ROW_SAMPLE, labels[train])
    predicted = svm.predict(dataset[test])[1]
    precision = accuracy_score(labels[test], predicted)
    precision_list.append(precision)
precision_cv_linear = np.mean(precision_list)

"""
#Cross Validation RBF kernel
rbf = []
for i in [0.1, 0.4, 0.7, 1]:
    
    kf = KFold(n_splits=10)
    precision_list = []
    for train, test in kf.split(dataset):
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_RBF)
        # svm.setDegree(0.0)
        svm.setGamma(0.1)
        # svm.setCoef0(0.0)
        # svm.setC(0)
        # svm.setNu(0.0)
        # svm.setP(0.0)
        # svm.setClassWeights(None)
        svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
        svm.train(dataset[train], cv2.ml.ROW_SAMPLE, labels[train])
        predicted = svm.predict(dataset[test])[1]
        precision = accuracy_score(labels[test], predicted)
        precision_list.append(precision)
    precision_cv_rbf = np.mean(precision_list)
    rbf.append([i, precision_cv_rbf])


#Cross Validation Polynomial kernel
poly = []
for i in [2, 3, 4]:
    kf = KFold(n_splits=10)
    precision_list = []
    for train, test in kf.split(dataset):
        svm = cv2.ml.SVM_create()
        svm.setType(cv2.ml.SVM_C_SVC)
        svm.setKernel(cv2.ml.SVM_POLY)
        svm.setDegree(2)
        svm.setGamma(1)
        svm.setCoef0(1)
        # svm.setC(0)
        # svm.setNu(0.0)
        # svm.setP(0.0)
        # svm.setClassWeights(None)
        svm.setTermCriteria((cv2.TERM_CRITERIA_COUNT, 100, 1.e-06))
        svm.train(dataset[train], cv2.ml.ROW_SAMPLE, labels[train])
        predicted = svm.predict(dataset[test])[1]
        precision = accuracy_score(labels[test], predicted)
        precision_list.append(precision)
    precision_cv_poly = np.mean(precision_list)
    poly.append([i, precision_cv_poly])
"""