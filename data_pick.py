####################
# to generate pkl data for training, first dump is for test in training with 100 samples,
#then each dump with 5000 samples for training
#x.pkl is curve points data
#y.pkl is ome, gamma data
#yp.pks is 10 p data


# coding: utf-8

# In[5]:


import matplotlib.pyplot as plt
import numpy as np
import random
import cPickle as pickle
import math

#general parameters
num_p=4  #num of states distributions
om0=100000   #om: 10k-100k(0-100k)
gamma0=10000  #gamma: 0-10k
num_points=1000
t=np.linspace(0,0.0003,num_points)   #time range, num of time points

sum_y=[]
fx=open('x.pkl','wb')
fy1=open('y.pkl','wb')
fy2=open('yp.pkl','wb')


#train data generate
batch_xs=[]
batch_ys=[]
batch_p_ys=[]
#size_test=100  #num of samples in test set
total_size=200000
pick_size=5000
for j in range (total_size):  #num of samples
    p=[]  #initialize outside the for loop and del inside the loop would be wrong
    om_r_set=[]  #set of omega and gamma
    sum_p=0
    sum_y=0  #final wave

    #om=random.uniform(0.8,1.2)
    #gamma=random.uniform(0.8,1.2)
    om=1
    gamma=1
    #om=j//(total_size/100)
    #om=om/10.0  #.0 is necessary
    
    #print(om)
    
    for i in range (0,num_p):
        p_i=random.uniform(0,10)
        p.append(p_i)
        sum_p=sum_p+p_i
    om_r_set.append(om) #normalized om(0-1) ~p
    om_r_set.append(gamma)
    #print(om_set)
    for i in range (0,num_p):
        p[i]=p[i]/sum_p     #normalize sum of p[i] to be 1
        Om=om*om0*math.sqrt(i+1)
        Gamma=gamma*gamma0*(i+1)**0.7
        y=p[i]*np.cos(2*Om*t)*np.exp(-Gamma*t)
        #om_i=math.sqrt(i) 
        #om_i=random.uniform(0,10)   # random omega
        #y=p[i]*np.cos(om_i*om*om0*t)*np.cos(om_i*om*om0*t)*np.exp(-gamma*gamma0*t)
        sum_y+=y
    ##some preprocessing
    #sum_y=np.fft.fft(sum_y)   #fft, use fft complex number as inputs of nn, it will discard imaginary part and maintain real part
    '''
    sum_y -= np.mean(sum_y, axis=0)  #zero-center
    sum_y /= np.std(sum_y, axis=0)   #normalize
    '''
    #plt.plot(t,sum_y)
    #plt.show()
    batch_xs.append(sum_y)
    batch_ys.append(om_r_set)
    batch_p_ys.append(p)
    if j % 10000 == 0:
        print(j)
    #first dump test in train data, size 100
    if j == 100:  
        pickle.dump(batch_xs,fx,True)
        pickle.dump(batch_ys,fy1,True)
        pickle.dump(batch_p_ys,fy2,True)
        print('dump_test in train:',j)

    #then dump train data
    if (j % pick_size == 0) and (j != 0):
        pickle.dump(batch_xs,fx,True)
        pickle.dump(batch_ys,fy1,True)
        pickle.dump(batch_p_ys,fy2,True)
        batch_xs=[]
        batch_ys=[]
        batch_p_ys=[]
        print('dump_train:', j)

fx.close()
fy1.close()
fy2.close()