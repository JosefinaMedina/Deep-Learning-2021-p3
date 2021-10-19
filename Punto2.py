import numpy as np
import grafica as gr
import math

#z =  6/(3*𝑥**2+2*𝑦**2+1);

[x, y, h] = gr.graficoGradiente(2)

# z = 3*x**2 + y**2
# PtoAnt = [x, y, z]
# x = x-1  
# y = y-2  #cambiamos x e y
# z = 3*x**2 + y**2;
# graficarPaso(PtoAnt, [x, y, z], h)

alfa = 0.2
MAX_ITE = 100  
ite = 1
z = 1
z_new = x**2*y*math.pi/3

while ((ite<MAX_ITE) and (math.fabs(z - z_new)>0.0001)):
    z = z_new
    PtoAnt = [x, y, z]
    grad_x = 2*x*y*math.pi/3   # derivada respecto de x
    grad_y = x**2*math.pi/3  # derivada respecto de y
    
    x = x - alfa * grad_x
    y = y - alfa * grad_y
    z_new = x**2*y*math.pi/3
    
    gr.graficarPaso(PtoAnt, [x, y, z_new], h)
    ite = ite + 1
    
print([x, y, z_new])