import numpy as np
import RN_feedforward as rn
# Ejemplos de entrada de la función AND
entradas = np.array([[0,0], [1,1],[0,1],[1,0]])
entradas = 2*entradas-1
salida = np.array([0,1,0,0])

# Tamaño de los datos de entrada y títulos
alfa = 0.1
MAX_ITE = 100
FUN = 'tansig'

if (FUN=='tansig'):
    salida = 2* np.array(salida * 1)-1  #lo convierte en [-1,1]
    
# Entrenamos la neurona no lineal para aproximar los valores de clase
[W, b, ite] = rn.entrena_NeuronaGradiente(entradas, salida, alfa, MAX_ITE, FUN, CotaError=10e-05, verIte=1, dibuja=1, titulos=['X1', 'X2'])

y_pred = rn.neurona_predice(entradas, W, b, FUN)

# netas = np.sum(entradas*W,axis=1)+b
# salidas = rn.evaluar(FUN,netas)
# if (FUN=='tansig'):
#     y_pred = 2*((salidas>0)*1)-1 
# if (FUN=='logsig'):
#     y_pred = (salidas>0.5)*1
    
    
print("%% aciertos: %.2f" % (100*np.sum(y_pred==salida)/entradas.shape[0]))
