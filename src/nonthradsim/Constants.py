import scipy.constants as C 

h = C.h/C.erg  # planck constant in erg*s
c = C.c*100    # speed of light in cm/s
q = 4.803e-10  # charge of electron in esu
sigma_T = 6.6524e-25  # Thomson cross-section in cm^2
erg2eV = 6.241509e11  # conversion factor from erg to eV
eV2erg = 1.602176634e-12  # conversion factor from eV to erg
me=  C.electron_mass * 1000  # electron mass in g