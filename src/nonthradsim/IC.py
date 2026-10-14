import numpy as np 
from scipy.special import kv
import nonthradsim.Constants as C
import matplotlib.pyplot as plt

def G(x_q,y_G):
    return 2*x_q*np.ln(x_q)+(1+2*x_q)*(1-x_q)+((1-x_q)*x_q**2*y_G**2)/(2*(1+y_G*x_q))

def IC(photon,particle):
    preE_ph=photon.E
    aftE_ph=photon.E
    preE_pa=particle.gm
    Gamma_e=4*preE_ph[None,None,:]*preE_pa[None,:,None]/(C.me*C.c**2) # Here Gamma is 2d array [E_photon, gm_particle]
    omega=aftE_ph[:, None,None]/(preE_pa[None,:,None]*C.me*C.c)
    q=omega/(Gamma_e*(1-omega))
    scat_cst=3*C.sigma_T* C.c/4
    dlogpreE= np.log(preE_ph[1:]/preE_ph[:-1]).mean()
    dloggm =np.log(preE_pa[1:]/preE_pa[:-1]).mean()
    dgm=preE_pa*dloggm

    P_intph=np.zeros_like(photon.value)
    for i in range(photon.n_spectrum):
        for j in range(particle.n_spectrum):
            P_intph+=scat_cst*photon.value[...,i,None]*dlogpreE*G(q,Gamma_e)[None,None,None,:,i,j]*particle.value[...,j,None]*dgm[j]/(preE_pa[j]**2)
    photon.value += P_intph
    return
    # P_intEph+=scat_cst*photon.value*dpreE
    # G_intE=G(q,Gamma_e).sum(0)
    # P_intE_add_par=(P_intEph*G_intE[None,None,None,:]/preE_pa**2)



    