import pandas as pd
import numpy as np
from grafica import *
import math

X = np.array([[0,0,0], [0,0,1], [0,1,0], [0,1,1], \
     [1,0,0], [1,0,1], [1,1,0], [1,1,1]])
Y = np.array([0,1,2,3,4,5,6,7]) 

cantEj = X.shape[0]  # cantidad de ejemplos de entrada
nAtrib = X.shape[1]
W = np.random.uniform(-0.5, 0.5, nAtrib)
b = np.random.uniform(-0.5, 0.5)
#b=0 #saca el bias
alfa = 0.1
MAX_ITE = 500 
ite = 1
E_ant = 1
E= np.mean((Y-(np.sum(X*W,axis=1)+b))**2)

while ((ite<MAX_ITE) and (math.fabs(E_ant - E)>10e-20)):
    E_ant=E
    for p in range(cantEj):

        salida = np.sum(X[p,:]*W) + b
        Error = Y[p]-salida
        
        grad_b = -2*Error
        grad_W = -2*Error*X[p, :]
    
        W = W - alfa * grad_W
        b = b - alfa * grad_b #comentar si b=0
        
    E= np.mean((Y-(np.sum(X*W,axis=1)+b))**2)

    ite = ite + 1
    print("ite = %d  -  error^2 = %.6f" % (ite, E))
print(W, b)