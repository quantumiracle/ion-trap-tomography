import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from IPython.display import display
from IPython.display import Math
import time
from scipy.optimize import curve_fit
from scipy.misc import factorial





class Optimizer(object):
    ## fit setting functions
    # multi parameter fitting 
    # https://stackoverflow.com/questions/34136737/using-scipy-curve-fit-for-a-variable-number-of-parameters




    def coherentPn(self,n,nbar):
        return nbar**n*np.exp(-nbar)/factorial(n)

    def thermalPn(n,nbar):
        return (nbar/(1.0+nbar))**n/(nbar+1)

    def FockDist(self, t, t0, Omega0, gamar0, Pn, N):
        n = np.arange(N).reshape(N,1)
        omega = Omega0 * np.sqrt(n+1)
        gamar_n = gamar0 * (n+1)**0.7
        Pn = Pn.reshape(N,1)
        #yn = Pn*np.sin(omega*(t-t0))**2*np.exp(-gamar_n*(t-t0))
        yn = Pn*np.cos(2*omega*(t-t0))*np.exp(-gamar_n*(t-t0))
        return np.sum(yn,axis=0)

    def CoherentDist(t, t0, Omega0, gamar0, nbar, N):
        n = np.arange(N).reshape(N,1)
        omega = Omega0 * np.sqrt(n+1)
        gamar_n = gamar0 * (n+1)**0.7
        Pn = coherentPn(n,nbar)
        yn = Pn*np.sin(omega*(t-t0))**2*np.exp(-gamar_n*(t-t0))
        return np.sum(yn,axis=0)

    # Pn distribution
    def wrapper_fit_func(self, t, N, *args):
        t0     = args[0][0]
        Omega0 = args[0][1]
        gamar0 = args[0][2]
        Pn = np.array(args[0][3:N+3])
        return self.FockDist(t, t0, Omega0, gamar0, Pn, N)

    #coherent distribution
    def wrapper_fit_coherent(t, N, *args):
        t0     = args[0][0]
        Omega0 = args[0][1]
        gamar0 = args[0][2]
        nbar = args[0][3]
        return CoherentDist(t, t0, Omega0, gamar0, nbar, N)



    def curve_fitter(self, dim, curve_set, curve_size, t_scale):
        t=np.linspace(0,t_scale,curve_size)
        y=curve_set
        N=dim
        nbar = 1.5
        #params_0 = [0.0, 2*np.pi/50, 0.0001]  # t0, omega0, gamar0
        params_0 += [self.coherentPn(i, nbar) for i in range(N)]  # using coherentstate distribution for initialization 
        plt.plot(t, y, 'o')
        #plt.show()
        popt, pcov = curve_fit(lambda t, *p_0: self.wrapper_fit_func(t, N, p_0), t, y, p0=params_0)  #*make use of parameter's dimension instead of num of parameters
        
        
        # plt.plot(t, wrapper_fit_func(t,N,params_0))
        plt.plot(t, self.wrapper_fit_func(t,N,popt),'r')
        plt.show()
        print(popt[3:])
        return(popt[3:])





class Optimizer_fix_omega(object):
    ## fit setting functions
    # multi parameter fitting 
    # https://stackoverflow.com/questions/34136737/using-scipy-curve-fit-for-a-variable-number-of-parameters

    def coherentPn(self,n,nbar):
        return nbar**n*np.exp(-nbar)/factorial(n)

    def thermalPn(n,nbar):
        return (nbar/(1.0+nbar))**n/(nbar+1)

    def FockDist(self, t, t0, gamar0, Pn, N, Omega0):
        n = np.arange(N).reshape(N,1)
        omega = Omega0 * np.sqrt(n+1)
        gamar_n = gamar0 * (n+1)**0.7
        Pn = Pn.reshape(N,1)
        #yn = Pn*np.sin(omega*(t-t0))**2*np.exp(-gamar_n*(t-t0))
        yn = Pn*np.cos(2*omega*(t-t0))*np.exp(-gamar_n*(t-t0))
        return np.sum(yn,axis=0)

    # Pn distribution
    def wrapper_fit_func(self, t, N, Omega0, *args):
        t0     = args[0][0]
        gamar0 = args[0][1]
        Pn = np.array(args[0][2:N+2])
        return self.FockDist(t, t0, gamar0, Pn, N, Omega0)


    def curve_fitter(self, om0, dim, curve_set, curve_size, t_scale):
        t=np.linspace(0,t_scale,curve_size)
        y=curve_set
        N=dim
        Omega0=om0
        nbar = 1.5
        params_0 = [0.0, 0.0001]  # t0, omega0, gamar0
        params_0 += [self.coherentPn(i, nbar) for i in range(N)]  # using coherentstate distribution for initialization 
        #origin curve points
        #plt.plot(t, y, 'o')
        #plt.show()
        popt, pcov = curve_fit(lambda t, *p_0: self.wrapper_fit_func(t, N, Omega0, p_0), t, y, p0=params_0)  #*make use of parameter's dimension instead of num of parameters
        
        
        # plt.plot(t, wrapper_fit_func(t,N,params_0))
        #fitted curve
        #plt.plot(t, self.wrapper_fit_func(t,N,Omega0,popt),'r')
        #plt.show()
        print(popt[2:])
        return(popt[2:])
