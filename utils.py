from tqdm import tqdm # for  well-established ProgressBar
from random import shuffle
import os #read or write a file
import cv2  
import numpy as np #supporting multi-dimensional arrays and matrices
from sklearn.model_selection import KFold
from itertools import product


#Funcion para etiquetar las imagenes
def label_img(word_label):                       
    if word_label == 'background': return 1.
    elif word_label == 'pedestrians': return 0.
    
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
        #data_lbp = [LBP_img_basic(img) for img in data]
        data_lbp = data.copy()
        for i in range(len(data)):
            data_lbp[i] = LBP_img_basic(data[i])
            print(i)
        return data_lbp


def LBP_img_basic(img):
    
    #Añadimos un zero padding
    size2, size1 = img.shape
    img_pad = np.zeros((size2 + 2, size1 + 2), dtype = 'int')
    img_pad[1:size2+1, 1:size1+1] = img
    
    #Desplazamiento y tamaño del bloque
    dx = 8
    dy = 8
    cell_x = 16
    cell_y = 16
    #Donde empieza nuestra ventana
    x = 1
    y = 1
    
    #Donde guardaremos los histogramas
    hist_list = []
    while  y + cell_y  <= size2+1:
        #Calculamos el valor de la LBP en cada pixel del bloque
        range_x = range(x, x + cell_x)
        range_y = range(y, y + cell_y)
        numbers_lbp = [lbp(i,j, img_pad) for i,j in product(range_y, range_x)]
        
        #Construimos el histograma
        hist = np.zeros(256)
        for l in numbers_lbp:
            hist[l] += 1
        hist_list = np.concatenate((hist_list, hist))
        
        #Seguimos moviendo el  bloque
        if x + dx + cell_x > size1+1:
            x = 1
            y = y + dy
        else:
            x = x + dx
    return hist_list

def lbp(i, j, img):
    """
    Calculamos el número LBP asociado
    """
    hood = img[i-1 : i+2, j-1:j+2]
    ordered_hood = np.concatenate((hood[0], [hood[1,2], hood[2,2], hood[2,1], hood[2,0], hood[1,0]]))
    ordered_hood_comparison = [0  if k < hood[1,1] else 1 for k in ordered_hood]
    binary = ""
    for digit in ordered_hood_comparison:
        binary += str(digit)
    integer = int(binary, 2)
    return integer