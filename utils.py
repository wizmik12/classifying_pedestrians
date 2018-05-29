from tqdm import tqdm # for  well-established ProgressBar
from random import shuffle
import os #read or write a file
import cv2  
import numpy as np #supporting multi-dimensional arrays and matrices
from sklearn.model_selection import KFold


#Funcion para etiquetar las imagenes
def label_img(word_label):                       
    if word_label == 'background': return 1.
    elif word_label == 'pedestrians': return 0.
#Nos devuelve la predicción del modelo
def label_return (model_out):
    if np.argmax(model_out) == 0: return  '0'
    elif np.argmax(model_out) == 1: return '1'
    
#Para crear el conjunto de train
def read_data(dir, CATEGORIES):
    data = []
    labels = []
    for category_id, category in enumerate(CATEGORIES):
        for img in tqdm(os.listdir(os.path.join(dir, category))):
            label=label_img(category)
            path=os.path.join(dir,category,img)
            img=cv2.imread(path, 0)
            data.append(img)
            labels.append(label)
    return data, labels


def hog_descriptor(data):
    """
    winSize = (64,128)
    blockSize = (16,16)
    blockStride = (8,8)
    cellSize = (8,8)
    nbins = 9
    derivAperture = 1
    winSigma = 4.
    histogramNormType = 0
    L2HysThreshold = 2.0000000000000001e-01
    gammaCorrection = 0
    nlevels = 64
    hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)
    
    
    winStride = (8,8)
    padding = (8,8)
    locations = ((10,20),)
    """
    hog = cv2.HOGDescriptor()
    for i in range(len(data)):
        #data[i] = hog.compute(data[i].astype('uint8'),winStride,padding,locations)
        data[i] = hog.compute(data[i].astype('uint8'))
    return data


def LBP(data):
        data_lbp = [LBP_img(img) for img in data]
        return data_lbp

def LBP_img(img):
    size1, size2 = img.shape
    numbers = []
    for i in range(1,size1-1):
        for j in range(1,size2-1):
            hood = img[i-1 : i+2,j-1:j+2]
            ordered_hood = np.concatenate((hood[0], [hood[2,1], hood[2,2], hood[2,1], hood[2,0], hood[1,0]]))
            for k in range(len(ordered_hood)):
                if ordered_hood[k] < hood [1,1]:
                    ordered_hood[k] = 0
                else:
                    ordered_hood[k] = 1
            
            binary = ""
            for digit in ordered_hood:
                binary += str(digit)
            integer = int(binary, 2)
            numbers.append(integer)
    
    hist = np.zeros(256)
    for l in numbers:
        hist[l] += 1
    return hist
