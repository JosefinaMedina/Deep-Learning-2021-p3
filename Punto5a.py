import numpy as np

def f(x):
    return 6*(x**2 + 1)

def fprima(x):
    return 12*x

alpha = 0.15
cota = 10e-05
xi= 10.
maxite = 100
ite =0

xeq =np.linspace(-3, 10, 1000)

#print(f(xi))
#temp1 = f(xi)
#temp2 = -fprima(xi)

    
for i in range(0, maxite):
#    global temp1, temp2, dif 
    temp1 = f(xi)
    temp2 = -fprima(xi) #el negativo porque nos interesa encontrar el minimo
    xi = -fprima(xi)*alpha + xi
    dif = abs(temp1 - f(xi))
    i = i +1
    ite = i
    print(i)
    if (dif < cota):
        print("Se llego a una diferencia menor a la tolerancia\nLa funcion evaluada en %.5f vale %.5f"%( xi, f(xi)))
        break

print("Se llego a %d iteraciones\nLa funcion evaluada en %.5f vale %.5f"%(ite, xi, f(xi)))