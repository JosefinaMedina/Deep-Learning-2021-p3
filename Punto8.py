import numpy as np
import RN_feedforward as rn

Ptos = np.array([[0,0,0,0], [0,0,1,1], [0,1,0,2], [0,1,1,3], [1,0,0,4],[1,0,1,5],[1,1,0,6],[1,1,1,7]])
X = Ptos[:,0:2]
Y = Ptos[:,3]

alfa = 0.01
MAX_ITE = 1000
Cota = 10e-06
dibuja=0
[W, b, ite] = rn.entrena_NeuronaLineal(X, Y, alfa, MAX_ITE, Cota,verIte=20, dibuja=0 )

print("Las iteraciones son: ", ite)
print("Los pesos son: ", W)
print("El bias es:", b)
#print("ite = %d   W = %.3f   b= %.3f" % (ite, W, b))

y_pred = rn.neurona_predice(X, W, b, 'purelin')
print("La y_pred es: ", y_pred)
ECM = np.mean((Y-y_pred)**2)
print("Error cuadratico promedio: %.6f" % ECM)