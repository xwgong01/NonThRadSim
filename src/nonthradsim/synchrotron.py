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

def synchrotron(photon, particle, magnetic_field, Fx):
    '''
    Photon, particle and magneticfield all contain 3d spatial information.
    The function will calculate the synchrotron spectrum for each spatial point.
    The photon spectrum will be updated in place.
    '''
    
    E_ph = photon.E # erg
    nu = E_ph / C.h
    gm = particle.gm + 1.
    B = magnetic_field
    if np.ndim(B) == 0:
        B = np.full(photon.grid.shape, B)
    N = particle.value
    dloggm = np.log(gm[1:]/gm[:-1]).mean()

    syn_spec = np.zeros_like(photon.value)
    for i in range(gm.shape[0]):
        nuc = 3 / 4 / np.pi * gm[i]**2 * C.q * B / C.me / C.c
        x = nu[None, None, None, :]/nuc[..., None]
        # print(B.shape, x.shape, N.shape)
        P = 3**0.5 * C.q **3 * B[..., None] / C.me / C.c**2 * np.interp(x,Fx[0],Fx[1])*N[..., i,None]
        # print(P.shape, gm[i].shape, dloggm)
        syn_spec += (P*gm[i]*dloggm)
        
    # print(syn_spec.max())
    photon.value += syn_spec
    return
