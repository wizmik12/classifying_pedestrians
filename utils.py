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
        data_lbp = [LBP_img_basic(img) for img in data]
        return data_lbp

"""
def LBP_img_basic(img):
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
"""
"""
def LBP_img_basic(img):
    size1, size2 = img.shape
    numbers = []
    despl_x = 8
    despl_y = 8
    celda_x = 16
    celda_y = 16
    x = 0
    y = 0
    
    hist_list = []
    while  y + celda_y < size1:
        for i in range(x, x + celda_x + 1):
            for j in range(y, y + celda_y + 1):
                
                if i == 0:
                    hood = img[i : i+2,j-1:j+2]
                    ordered_hood = np.concatenate((hood[0], [hood[1,2], hood[2,2], hood[2,1]]))
                    
                elif j == 0:
                    hood = img[i-1 : i+2,j:j+2]
                    ordered_hood = np.concatenate(([hood[1,2], hood[2,2], hood[2,1], hood[2,0], hood[1,0]]))
                    
                elif j == 0 and i == 0:
                    hood = img[i : i+2,j:j+2]
                    ordered_hood = np.concatenate((hood[0], [hood[1,2], hood[2,2], hood[2,1], hood[2,0], hood[1,0]]))
                    
                elif i == size2:
                    hood = img[i : i+1,j:j+2]
                    ordered_hood = np.concatenate((hood[0], [hood[1,2], hood[2,2], hood[2,1], hood[2,0], hood[1,0]]))
                    
                elif j == size1:
                    hood = img[i : i+2, j:j+1]
                    ordered_hood = np.concatenate((hood[0], [hood[1,2], hood[2,2], hood[2,1], hood[2,0], hood[1,0]]))
                    
               elif i == size2 and j == size1:
                    hood = img[i : i+1,j:j+1]
                    ordered_hood = np.concatenate((hood[0], [hood[1,2], hood[2,2], hood[2,1], hood[2,0], hood[1,0]]))
                    
                else:
                    hood = img[i-1 : i+2,j-1:j+2]
                    ordered_hood = np.concatenate((hood[0], [hood[1,2], hood[2,2], hood[2,1], hood[2,0], hood[1,0]]))
                    
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
        hist_list.append(hist)
        
        if x + despl_x + celda_x > size2:
            x = 0
            y = y + despl_y
        else:
            x = x + despl_x
            
    return hist_list
"""


def LBP_img_basic(img):
    size2, size1 = img.shape
    numbers = []
    despl_x = 8
    despl_y = 8
    celda_x = 16
    celda_y = 16
    x = 0
    y = 0
    
    hist_list = []
    while  y + celda_y  <= size2:
        for i in range(y, y + celda_y):
            for j in range(x, x + celda_x):
                
                hood = np.zeros((3,3), dtype = int)
                
                if j == 0 and i == 0:
                    hood[1:3, 1:3] = img[i:i+2, j:j+2]
                 
                elif i == 0 and j == size1 - 1:
                    hood[1:3, 0:2] = img[i:i+2, j-1:j+1]
                    
                elif j == 0 and i == size2 - 1:
                    hood[0:2, 1:3] = img[i-1:i+1, j:j+2]
                    
                elif i == size2 - 1 and j == size1 - 1:
                    hood[0:2, 0:2] = img[i-1 : i+1, j-1:j+1]
                    
                elif i == 0:
                    hood[1:3,0:3] = img[i:i+2, j-1:j+2]
                            
                elif j == 0:
                    hood[0:3, 1:3] = img[i-1:i+2, j:j+2]
                                      
                
                    
                elif i == size2 - 1:
                    hood[0:2, 0:3] = img[i-1 : i+1, j-1:j+2]
                    
                    
                elif j == size1 - 1:
                    hood[0:3, 0:2] = img[i-1 : i+2, j-1:j+1]
                    
                else:
                    hood = img[i-1 : i+2, j-1:j+2]
                    
                
                
                print(j,i,x,y)
                ordered_hood = np.concatenate((hood[0], [hood[1,2], hood[2,2], hood[2,1], hood[2,0], hood[1,0]]))
                    
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
        hist_list = np.concatenate((hist_list, hist))
        
        if x + despl_x + celda_x > size1:
            x = 0
            y = y + despl_y
        else:
            x = x + despl_x
    return hist_list


"""
%Función para calcular el histograma de orientaciones del gradiente
function H=HOG(Im)
nwin_x=3;%número de ventanas por cajas
nwin_y=3;
B=9;%El número de intervalos del histograma (partiremos en 9 trozos los 180 grados)
[L,C]=size(Im); % L número de líneas ; C número de comlunas
H=zeros(nwin_x*nwin_y*B,1); % vector columna con ceros
m=sqrt(L/2);
if C==1 % si el número de columnas es cero
    Im=im_recover(Im,m,2*m);%verifica el tamaño de la imagen
    L=2*m;
    C=m;
end
Im=double(Im);
step_x=floor(C/(nwin_x+1));
step_y=floor(L/(nwin_y+1));
cont=0;
hx = [-1,0,1]; %kernel para filtrar la imagen y obtener el gradiente horizontal
hy = -hx'; %kernel para filtrar la imagen y obtener el gradiente vertical
grad_xr = imfilter(double(Im),hx); %filtramos la imagen (gradiente horizontal)
grad_yu = imfilter(double(Im),hy); %gradiente vertical
angles=atan2(grad_yu,grad_xr); %angulo del gradiente en cada coordenada
magnit=((grad_yu.^2)+(grad_xr.^2)).^.5; %obtenemos la magnitud del gradiente
for n=0:nwin_y-1
    for m=0:nwin_x-1
        cont=cont+1; %actualizamos el contador
        angles2=angles(n*step_y+1:(n+2)*step_y,m*step_x+1:(m+2)*step_x);  
        magnit2=magnit(n*step_y+1:(n+2)*step_y,m*step_x+1:(m+2)*step_x);
        v_angles=angles2(:);    
        v_magnit=magnit2(:);
        K=max(size(v_angles)); 
        %Montamos el histograma con 9 intervalos (rango de 20 grados por intervalo)
        bin=0;
        H2=zeros(B,1);
        for ang_lim=-pi+2*pi/B:2*pi/B:pi %recorremos los ángulos
            bin=bin+1;
            for k=1:K
                if v_angles(k)<ang_lim
                    v_angles(k)=100; %Ponemos a 100 el ángulo para que no lo vea más
                    H2(bin)=H2(bin)+v_magnit(k); %Sumamos el valor de la magnitud de ese ángulo
                end
            end
        end
                
        H2=H2/(norm(H2)+0.01);     %Normalizamos el vector    
        H((cont-1)*B+1:cont*B,1)=H2;
    end
end
"""