import os
import pandas as pd
import numpy as np
import RN_feedforward as rn
from sklearn import preprocessing
from sklearn.metrics import mean_squared_error
    
# Leer FrutasTrain.csv
os.chdir('../03_CombinadorLineal/')
datos = pd.read_csv("AUTOS.csv")

#Completar los datos faltantes de price
meanprice = datos[' price'].mean()

#aplicar el reemplazo
datos[' price'] = datos[' price'].replace([np.nan], meanprice)

X = np.array(datos.iloc[:,-11]).reshape(-1,1)
Y = np.array(datos.iloc[:,-2]).reshape(-1,1)

normalizarEntrada = 1  # 1 si normaliza; 0 si no
if normalizarEntrada:
    # Escala los valores entre 0 y 1
    # normalizador = preprocessing.MinMaxScaler()
    
    # Normaliza utilizando la media y el desvio
    normalizador= preprocessing.StandardScaler()
    X = normalizador.fit_transform(X)
    Y = normalizador.fit_transform(Y)

alfa = 0.001
MAX_ITE= 500
Cota = 10e-10
dibuja = 0
titulos = ['X', 'Y']
verIte = 20

[W, b, ite] = rn.entrena_NeuronaLineal(X, Y, alfa, MAX_ITE,Cota, verIte, dibuja,titulos)
yTrain=rn.neurona_predice(X,W,b,'purelin')
print("ite = %d   W = %.3f   b= %.3f" % (ite, W, b))
y_pred=(W * X + b)
y_pred2 = rn.neurona_predice(X, W, b, 'purelin')
ECM = np.mean((Y-y_pred)**2)

# Error Cuadrado Medio
print("Mean squared error: %.6f" % mean_squared_error(Y, y_pred))
