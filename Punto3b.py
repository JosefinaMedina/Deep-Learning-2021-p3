import numpy as np
import grafica as gr
import math

#z =  6/(3*𝑥**2+2*𝑦**2+1);

[x, y, h] = gr.graficoGradiente(3)

#x=0.5
#y=-0.5 #no encuentra el minimo, pocos pasos #0.36479147582461613, -0.36479147582461613, -2.369395669242615

x=-2
y=2 #encuentra el minimo #-0.007386813429733997, 0.007386813429733997, -2.9996726456483147

#x=1.5
#y=1.5 #encuentra el minimo #0.3850137412134155, -0.2766026524119877, -2.449490269077474

alfa = 0.05
MAX_ITE = 100  
ite = 1
z = 1
z_new = -3/(x**2 + y**2 + 1)

while ((ite<MAX_ITE) and (math.fabs(z - z_new)>0.000001)):
    z = z_new
    PtoAnt = [x, y, z]
    grad_x =  6*x/(x**2+y**2 +1)**2   # derivada respecto de x
    grad_y =  6*y/(x**2+y**2 +1)**2  # derivada respecto de y
    
    x = x - alfa * grad_x
    y = y - alfa * grad_y
    z_new = -3/(x**2 + y**2 + 1)
    
    gr.graficarPaso(PtoAnt, [x, y, z_new], h)
    ite = ite + 1
    
print(PtoAnt, [x, y, z_new], h)