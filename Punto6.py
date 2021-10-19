import os
import pandas as pd
import numpy as np
import RN_feedforward as rn
    
# Leer 
os.chdir('../03_CombinadorLineal/')
datos = pd.read_csv("CCPP.csv")

#--- EJEMPLO DE ENTRENAMIENTO ---
X = np.array(datos.iloc[:,0])
Y = np.array(datos.iloc[:,-1])

print(len(X))

W1 = -1.89  
b1= 498.22
W2 = -1.92 
b2= 497.43
W3 = -1.07 
b3= 517.92
W4 = -1.09 
b4 = 512.98
W5 = -2.02 
b5= 503.11
#W1 = -2.00
#b1= 500
#W2 = -2.21  
#b2= 498
#W3 = -2.30 
#b3= 497
#W4 = -2.22 
#b4=496.5
#W5 = -2.16  
#b5=496.91
#N=9568

y_pred1=(W1 * X + b1)
ECM1 = np.mean((Y-y_pred1)**2)
print("Error cuadratico promedio 1:", ECM1)
y_pred2=(W2 * X + b2)
ECM2 = np.mean((Y-y_pred2)**2)
print("Error cuadratico promedio 2:", ECM2)
y_pred3=(W3 * X + b3)
ECM3 = np.mean((Y-y_pred3)**2)
print("Error cuadratico promedio 3:", ECM3)
y_pred4=(W4 * X + b4)
ECM4 = np.mean((Y-y_pred4)**2)
print("Error cuadratico promedio 4:",  ECM4)
y_pred5=(W5 * X + b5)
ECM5 = np.mean((Y-y_pred5)**2)
print("Error cuadratico promedio 5:",  ECM5)

#el 2 y el 5 dan con el menor error