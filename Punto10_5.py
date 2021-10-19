import pandas as pd
import numpy as np
from sklearn import model_selection,preprocessing
import RN_feedforward as rn

datos = pd.read_csv('../03_CombinadorLineal/Iris.csv')
entradas = np.array(datos.iloc[:,:-1])   #-- todas las columnas menos la última


#--- SALIDA BINARIA : 1 si es "drugY" ; 0 si no ---
salidas = np.array(datos['class']=="Iris-setosa") * 1
nomClase = ['Otra', 'Iris-setosa']
#--- CONJUNTOS DE ENTRENAMIENTO Y TESTEO ---
X_train, X_test, T_train, T_test = model_selection.train_test_split(
        entradas, salidas, test_size=0.30, random_state=42)

normalizarEntrada = 1

if normalizarEntrada:
    # Escala los valores entre 0 y 1
    normalizador = preprocessing.MinMaxScaler()
    entradas = normalizador.fit_transform(entradas)
        
salida = salidas[orden]
entradas = entradas[orden, :]

i=0
for i in range(10):
   
    alfa = 0.2
    MAX_ITE = 500
    FUN = 'tansig'
    if (FUN=='tansig'):
        salida = 2* np.array(salida * 1)-1  #lo convierte en [-1,1]
        
        # Entrenamos la neurona no lineal para aproximar los valores de clase
        [W, b, ite] = rn.entrena_NeuronaGradiente(entradas, salida, alfa, MAX_ITE, FUN, CotaError=10e-05, verIte=1, dibuja=1, titulos=['Diametro', 'Color'])
        
        netas = np.sum(entradas*W,axis=1)+b
        salidas = rn.evaluar(FUN,netas)
        if (FUN=='tansig'):
            y_pred1 = 2*((salidas>0)*1)-1 
            if (FUN=='logsig'):
                y_pred1 = (salidas>0.5)*1

    y_pred = rn.neurona_predice(entradas, W, b, FUN)

print("cantidad aciertos Train: %d" % np.sum(y_pred==salida))
# -- La neurona no lineal ya está entrenada ---
# Leer FrutasTest.csv y ver que responde el perceptrón
if (FUN=='tansig'):
    T_test = 2* np.array(T_test * 1)-1  #lo convierte en [-1,1]

datosTest = pd.read_csv("FrutasTest.csv")
salidaTest = T_test
salidaTest = datosTest['Clase']==nomClase[1]
X_Test = np.array(datosTest.iloc[:,0:-1])
if normalizarEntrada:
    X_Test = normalizador.transform(X_test)
# Calcular las netas de los ejemplos de testeo
netasTest = np.sum(X_test*W,axis=1)+b
salidasTest = rn.evaluar(FUN,netasTest)
if (FUN=='tansig'):
    y_predTest1 = 2*((salidasTest>0)*1)-1 
if (FUN=='logsig'):
    y_predTest1 = (salidasTest>0.5)*1

y_predTest = rn.neurona_predice(X_test, W, b, FUN)
print("cantidad aciertos Test: %d " % np.sum(y_predTest))