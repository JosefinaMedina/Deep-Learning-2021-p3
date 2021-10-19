import os
import pandas as pd
import numpy as np
import RN_feedforward as rn
from sklearn import preprocessing, model_selection
    
# Leer Autos.csv
os.chdir(r'C:\Users\Josefina Medina\Documents\FISICA FACULTAD\CUARTO\deep learning\03_CombinadorLineal')
#os.chdir('../03_CombinadorLineal/')
datos = pd.read_csv("..//03_CombinadorLineal/Vinos.csv")
ncolum= datos.columns.values

print(datos.isnull().sum())

entradas = np.array(datos.iloc[:,0:11])
salidas = np.array((datos['Class']== 1)*1)
nomClase = ['Otra', '1']


X_train, X_test, T_train, T_test = model_selection.train_test_split(
        entradas, salidas, test_size=0.20, random_state=42)


#--- SALIDA BINARIA : 1 si es "drugY" ; 0 si no ---
#salidas = np.array(datos[]) * 1
#nomClase = ['Otra', 'Iris-setosa']


normalizarEntrada = 1 # 1 si normaliza; 0 si no
if normalizarEntrada:
    # Escala los valores entre 0 y 1
    min_max_scaler = preprocessing.MinMaxScaler()
    X_train = min_max_scaler.fit_transform(X_train)
    X_test = min_max_scaler.transform(X_test)


FUN = 'logsig'

alfa = 0.05
MAX_ITE = 150
[W, b, ite] = rn.entrena_NeuronaGradiente(X_train, T_train, alfa, MAX_ITE, FUN, CotaError=10e-10, verIte=20)

#netas = np.sum(entradas*W,axis=1)+b
#salidas = rn.evaluar(FUN,netas)
#if (FUN=='tansig'):
#    y_pred1 = 2*((salidas>0)*1)-1 
#if (FUN=='logsig'):
#    y_pred1 = (salidas>0.5)*1

y_pred = rn.neurona_predice(X_test, W, b, FUN )
#y_train = rn.neurona_predice(X_train, W, b, FUN )
print("La y_pred es: ", y_pred)
#ECM = np.mean(( T_test -y_pred)**2)
Aciertospor =100*np.sum(T_test==y_pred)/len(T_test)
print("Porcentaje de acierto: %.6f %%" % Aciertospor)

# Calcular las respuestas del perceptron
#yTest = rn.aplica_Perceptron(X_test,W,b)



#aciertosTrain = 100 * np.sum(T_test==T_train)/len(T_train) #T_test o what?

#aciertosTest  = 100 * np.sum(yTest==T_test)/len(T_test)
#print("iteraciones utilizadas = ",ite)
#print("%% aciertos datos entrenamiento %.2f:" % aciertosTrain)
#print("%% aciertos datos de testeo %.2f:" % aciertosTest)