import numpy as np
import grafica as gr
import RN_feedforward as rn

# Ejemplos de entrada de la función AND
X = np.array([[0,0], [1,1],[0,1],[1,0]])
X = 2*X-1
T = np.array([0,1,0,0])

# Tamaño de los datos de entrada y títulos
alfa = 0.1
MAX_ITE = 100
FUN = 'logsig'
if (FUN=='tansig'):
    T = 2* np.array(T * 1)-1  #lo convierte en [-1,1]

# Tamaño de los datos de entrada y títulos
nCantEjemplos = X.shape[0]  # nro. de filas
nAtrib = X.shape[1]         #nro. de columnas
titulos = ['X1', 'X2']

# Inicializar la recta
W = np.array(np.random.uniform(-0.5, 0.5, size=2))
b = np.random.uniform(-0.5, 0.5)

# graficar
gr.dibuPtos(X, T, titulos)
ph = gr.dibuRecta(X, W, b)

MAX_ITE = 300
alfa = 0.1
ite=0
CotaError = 10e-04
AVGError = 1
AVGErrorAnt=0

while ((ite<MAX_ITE) and (abs(AVGErrorAnt-AVGError)>CotaError)):
    AVGErrorAnt = AVGError
    suma= 0.0
    for e in range(nCantEjemplos):
        # Calcular y  (la salida de la neurona no lineal)
        neta = b + W[0]*X[e,0] + W[1]*X[e,1]
        y = rn.evaluar(FUN,neta)
        # -- error en este ejemplo ---
        errorK = T[e]-y
       
        W = W + alfa * errorK * rn.evaluarDerivada(FUN,y)*X[e,:]
        b = b + alfa * errorK * rn.evaluarDerivada(FUN,y)
        
        #-- acumular el error al cuadrado para promediar al final
        suma = suma + errorK**2
    # graficar la recta
    ph = gr.dibuRecta(X, W, b, ph)
    
    # --- aprox. del error cuadrático promedio ---    
    AVGError = suma/nCantEjemplos   
    
    #--- ECM calculado sobre los ejemplos
    netas = np.sum(X*W, axis=1)+b
    salidas = rn.evaluar(FUN, netas)
    AVGError2 = np.mean((T-salidas)**2)
    
    ite = ite + 1
    print("ite %d   -  ECM (aprox) = %.5f    ECM = %.5f" % (ite, AVGError, AVGError2))

y_pred = rn.neurona_predice(X, W, b, FUN)

netas = np.sum(X*W,axis=1)+b
salidas = rn.evaluar(FUN,netas)
if (FUN=='tansig'):
    y_pred = 2*((salidas>0)*1)-1 
if (FUN=='logsig'):
    y_pred = (salidas>0.5)*1
    
    
print("%% aciertos: %.2f" % (100*np.sum(y_pred==T)/X.shape[0]))
    