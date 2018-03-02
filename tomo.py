from sympy import *
from qutip import *
import numpy as np
import qinfer as qi
import matplotlib.pyplot as plt
import math
from measure import Measure
from NNpredictor import Predictor

dim = 4
num_dis = 12  #number of displacements

om0=100000   #om: 10k-100k(0-100k)
gamma0=10000  #gamma: 0-10k
num_points=1000
t_scale=0.0003


measure = Measure(dim = dim, num_dis = num_dis)
example_matrix = measure.example_matrix()
print(example_matrix)
alpha, dis, dis_bas, P = measure.measure()
print(P)


t=np.linspace(0,t_scale,num_points) 
curve_set=measure.generate_curves(om0, gamma0, num_points, t_scale)
#show generated curves
for j in range (num_dis):
    #print(P[j],alpha[j])
    plt.plot(t,curve_set[j])
    #plt.show()

#density matrix tomograph with original P
tomo_matrix = measure.tomograph()
#print(tomo_matrix, example_matrix)

#density matrix tomograph with predicted P
predictor = Predictor(curve_set)
predict_P = predictor.NNpredictor()
print(P,'\n',predict_P)
predict_tomo_matrix = measure.predict_tomograph(predict_P)
print(tomo_matrix, predict_tomo_matrix, example_matrix)