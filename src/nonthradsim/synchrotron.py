from nonthradsim.spectrum import Spectrum
import numpy as np 
from scipy.special import kv
import nonthradsim.Constants as C
import matplotlib.pyplot as plt

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
    numx = 1001
    x =np.logspace(-5,2,numx)
    
    xx, XX = np.meshgrid(x,x)
    Integrand = XX * K(xx)
    Integrand = np.where(xx>=XX,Integrand,0)
    Fx = ((Integrand[:,1:]+Integrand[:,:-1])/2*(x[1:]-x[:-1])).sum(1)
    return x, Fx

def synchrotron(photon_spec, particle_spec, magnetic_field, Fx):
    
    E_ph = photon_spec.E # erg
    nu = E_ph / C.h
    gm = particle_spec.E + 1
    B = magnetic_field
    N = particle_spec.spec

    nu,gm = np.meshgrid(nu,gm)
    nuc = 3 / 4 / np.pi * gm**2 * C.q * B / C.me / C.c
    # imax =  np.argmax(N)
    # print('nuc :', nuc[imax,0])
    x = nu/nuc
    P = 3**0.5 * C.q **3 * B / C.me / C.c**2 * np.interp(x,Fx[0],Fx[1])*N[:, np.newaxis]
    syn_spec = ((P[1:,:]+P[:-1,:])/2*(gm[1:,:]-gm[:-1,:])).sum(0)
    # print(syn_spec.max())
    photon_spec.spec += syn_spec
    return