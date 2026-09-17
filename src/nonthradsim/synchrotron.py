from .spectrum import Spectrum
import numpy as np 
from scipy.special import kv
import math

def K(x):
    return kv(5/3,x)

def F():
    r"""
    Args: 
    
    Returns:
    1. Fx, tuple,Fx = (x,F(x))
    
    .. math::
        
        F(\xi) = x\int_x^\infty K_{5/3}(\xi)d\xi
    
    """
    x =np.logspace(-100,3,3001)
    result = np.zeros_like(x)
    numx = x.shape[0]
    X = np.logspace(-100,3,1000)
    for i in range(numx):
        if x[i]<=1e-100:
            result[i] = 0
        else:
            ind_x = np.argmin(np.abs(X-x[i]))+1

            func_value = X[ind_x-1] * ((K(X[ind_x+1:])
                                      +
                                      K(X[ind_x-1:-2])
                                     )/2
                                    * (X[ind_x+1:]-X[ind_x:-1])
                                    ).sum()
            result[i] = func_value
    return (x,result)

def synchrotron_prof(nu,gm,m,N,B,Fx):
    '''
    Args:
    1. nu, freq of radiation
    2. gm, lorentz factor of electron
    3. m, mass of electron
    4. N, num density of electron
    5. B, magnetic field
    6. Fx, the function value table calculated in F()
    
    Returns:
    1. syn_spec, erg/s/erg, dP/dE
    '''
    h = C.h/C.erg
    c = C.c*100
    q = 4.803e-10
    nu,gm = np.meshgrid(nu,gm)
    nuc = 3 / 4/np.pi * gm**2 * q * B / m / c
    N = N.reshape(-1,1)
    x = nu/nuc
    P = 3**0.5 * q **3 * B / m / c**2 * np.interp(x,Fx[0],Fx[1])*N
    syn_spec = ((P[1:,:]+P[:-1,:])/2*(gm[1:,:]-gm[:-1,:])).sum(0)
    return syn_spec/h



def synchrotron(nu, gme, m, Ne, B):
    '''implicit none
    integer*4 i
    real*8 pi, m, c, e, h, sigmaTh,R
    real*8 B, gme(NumEe), Ne(NumEe), nuc(NumEe), syn_spec(NumEg), nu(NumEg), syn1(NumEg), syn2(NumEg)
    real*8 a1(3), a2(3), H1(NumEg), H2(NumEg), F1(NumEg), F2(NumEg), F(NumEg), x(NumEg), synchrotron(NumEg)'''

    e=4.8e-10
    pi=3.1416e0
    h=6.63e-27
    c=2.998e10

    a1=[-0.97947839e0, -0.83333239e0, 0.15541796e0]
    a2=[-4.69247166e-2, -0.700510018e0, 1.03876298e-2]


    nuc=3e0/4/pi*gme**2*e*B/m/c
    syn_spec=0
    NUM_E = gme.shape[0]
    for i in range(0, NUM_E-1):
        x=nu/nuc[i]
        F1=pi*2**(5./3)/3**0.5/math.gamma(1./3)*x**(1./3)
        F2=(pi/2)**0.5*np.exp(-x)*x**0.5
        H1=a1[0]*x+a1[1]*x**(1./2)+a1[2]*x**(1./3)
        H2=a2[0]*x+a2[1]*x**(1./2)+a2[2]*x**(1./3)
        F=F1*np.exp(H1)+F2*(1-np.exp(H2))

        syn1=F*3**0.5*e**3*B/m/c**2  # 这里的syn1是dP/d gm

        x=nu/nuc[i+1]
        F1=pi*2**(5./3)/3**0.5/math.gamma(1./3)*x**(1./3)
        F2=(pi/2)**0.5*np.exp(-x)*x**0.5
        H1=a1[0]*x+a1[1]*x**(1./2)+a1[2]*x**(1./3)
        H2=a2[0]*x+a2[1]*x**(1./2)+a2[2]*x**(1./3)
        F=F1*np.exp(H1)+F2*(1-np.exp(H2))

        syn2=F*3**0.5*e**3*B/m/c**2

        syn_spec=syn_spec+(syn1*Ne[i]+syn2*Ne[i+1])*(gme[i+1]-gme[i])/2 # 用梯形法则积分，得到P，也就是所有粒子的总功率，单位是erg/s
        #print(syn_spec.shape)
    return syn_spec/h

if __name__ == '__main__':
    from time import time
    h = C.physical_constants['Planck constant in eV/Hz'][0]*C.eV/C.erg
    m = C.electron_mass *1000
    c = C.c*100
    gm = np.logspace(1,10,211)
    nu = np.logspace(-10,49,38)
    Fx = F()
    B = 0.07
    n = gm**(-2)*1e20
    #n = np.where((gm < 9e5) | (gm > 1e6),0,n)
    tbeg = time()
    spec = synchrotron(nu,gm,m,n,B,Fx)
    print('mytime %.5f'%(time()-tbeg))
    
    

    
    tbeg = time()
    spec2 = synchrotron_prof(nu,gm,m,n,B)
    print('proftime %.5f'%(time()-tbeg))
    
    
    plt.loglog(nu*h*C.erg/C.eV,spec)
    plt.loglog(nu*h*C.erg/C.eV,spec2)
    plt.ylim(1e-10,1e01)
    plt.show()
    
    
    Elosstot = ((spec[1:]+spec[:-1])/2*(nu[1:]-nu[:-1])).sum()
    print('h = %.2e'%h)
    
    r0 = 2.817940327e-13
    sigT = 8 * np.pi / 3 * r0**2
    Elosstrue = 2*sigT*c *gm**2* B**2/8/np.pi
    E = gm * m * c**2 #erg
    Elosstot_true = ((Elosstrue[1:]+Elosstrue[:-1])/2*(gm[1:]-gm[:-1])*(n[1:]+n[:-1])/2).sum()
    print(Elosstot/Elosstot_true)