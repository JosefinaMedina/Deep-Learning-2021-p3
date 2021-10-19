import numpy as np
from matplotlib import pyplot as plt
from grafica import *
import sys

ptos = np.array([[-1,3],[3,1],[1,0],[3,3],[0,1],[2,4],[-1,1],[2,5]])

#T2 = np.array([ -1, 1, -1, 1, -1, 1,  -1, 1 ])

T2 = np.array([ 0, 1, 0, 1, 0, 1,  0, 1 ])

W = np.random.uniform(-0.5,0.5, 2)  #[0.7949,  0.3120]
b = np.random.uniform(-0.5,0.5, 1)  #-2.0388
FUN = 'logsig'

[ph, h] = graficarFuncionActivacion(ptos, T2, W, b, FUN)
MAX_ITE = 100
alfa = 0.25
ite = 0
E_ant =0
Error = 1

while (ite<MAX_ITE): # and (math.fabs(E_ant-Error)>0.000001):
    E_ant = Error
    for p in range(len(ptos)):

        neta = W[0]*ptos[p,0]+W[1]*ptos[p,1]+b

        y = evaluar(FUN, neta)
        
        Error = T2[p]-y
        gradiente_W = -2 * Error * evaluarDerivada(FUN,y) * ptos[p,:]
        gradiente_b = -2 * Error * evaluarDerivada(FUN,y)
        
        W = W - alfa * gradiente_W
        b = b - alfa * gradiente_b
    
    [ph, h] = graficarFuncionActivacion(ptos, T2, W, b, FUN, ph, h)

    ite = ite + 1
    
neta = np.sum(np.outer(np.ones(len(ptos)),W)*ptos,axis=1)+b
y = evaluar(FUN, neta)
np.savetxt(sys.stdout, y, "%.5f")  






    
