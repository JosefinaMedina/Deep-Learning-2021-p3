import os
import pandas as pd
import numpy as np
import RN_feedforward as rn
from matplotlib import pyplot as plt
from sklearn import preprocessing
import chardet
    
# Leer FrutasTrain.csv
os.chdir('../03_CombinadorLineal/')
nomArch = "AUTOS1.csv"

with open(nomArch, 'rb') as f:
    result=chardet.detect(f.read())

df=pd.read_csv(nomArch, encoding=result['encoding'])

#Completar los datos faltantes de price

meanprice = df[' price'].mean()
meanhorsepw = df[' horsepower'].mean()
meanbore = df[' bore'].mean()
meanstroke = df[' stroke'].mean()
meanprice = df[' price'].mean()
meanpeak= df[' peak-rpm'].mean()
meannormloss=df['normalized-losses'].mean()

#aplicar el reemplazo
df[' price'] = df[' price'].replace([np.nan], meanprice)
df[' horsepower'] = df[' horsepower'].replace([np.nan], meanhorsepw)
df[' bore'] = df[' bore'].replace([np.nan], meanbore)
df[' stroke'] = df[' stroke'].replace([np.nan], meanstroke)
df[' price'] = df[' price'].replace([np.nan], meanprice)
df[' peak-rpm'] = df[' peak-rpm'].replace([np.nan], meanpeak)
df['normalized-losses']=df['normalized-losses'].replace([np.nan],meannormloss)
print(df.isnull().sum())
nomCol=np.array(df.columns.values).reshape(-1,1)
mat= df.corr()

Ejemplos= np.array(df)
normalizarEntrada= 1
if normalizarEntrada:
    # Escala los valores entre 0 y 1
    normalizador = preprocessing.MinMaxScaler()
    Ejemplos = normalizador.fit_transform(Ejemplos)
    
attPred=14
T=Ejemplos[:,attPred].reshape(-1,1)
X=np.delete(Ejemplos,[attPred], 1)
nomBuscado= nomCol[attPred]
elegidos=[]
itera=[]
nAtrib=X.shape[1]
W_Obtenidos=np.zeros((30,nAtrib))
Votos_pos = np.zeros(nAtrib)
Votos_neg = np.zeros(nAtrib)
alfa = 0.01
MAX_ITE=1000
CotaError= 10e-08
verIte = 0

for intentos in range(30):
    [W, b, ite] = rn.entrena_NeuronaLineal(X, T, alfa, MAX_ITE,CotaError, verIte)
    yTrain=rn.neurona_predice(X,W,b,'purelin')

    itera.append(ite)
    ordenElecc = np.argsort(-1*W).tolist()
    positivos=1*(W[ordenElecc[0:2]]>0)
    negativos=1*(W[ordenElecc[nAtrib-2:nAtrib]]<0)
    Votos_pos[ordenElecc[0:2]]=Votos_pos[ordenElecc[0:2]]+positivos
    #Votos_pos[ordenElecc[nAtrib-2:nAtrib]]=Votos_pos[ordenElecc[nAtrib]]
    elegidos.append(ordenElecc)
    W_Obtenidos[intentos, :]=W
print("alfa %.2f, MAX_ITE %d, AVG Ite = %2f" %(alfa, MAX_ITE,np.mean(itera)))

barras=Votos_pos
plt.figure()
N=len(barras)
plt.bar(np.arange(N), barras)
plt.title(nomBuscado)
plt.ylabel('atributos usados')
plt.xticks(np.arange(N), nomCol, rotation='vertical')
plt.subplots_adjust(bottom=0.3)
plt.show()

barras=W
plt.figure()
N=len(barras)
plt.bar(np.arange(N), barras)
plt.title(nomBuscado)
plt.ylabel('W')
plt.xticks(np.arange(N), nomCol, rotation='vertical')
plt.subplots_adjust(bottom=0.3)
plt.show()

barras= np.mean(W_Obtenidos,axis=0)
plt.figure()
N=len(barras)
plt.bar(np.arange(N), barras)
plt.title(nomBuscado)
plt.ylabel('W promedio')
plt.xticks(np.arange(N), nomCol, rotation='vertical')
plt.subplots_adjust(bottom=0.3)
plt.show()
