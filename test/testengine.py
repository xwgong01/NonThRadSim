from nonthradsim.grid import *
from nonthradsim.particle import *
from nonthradsim.spectrum import *
from nonthradsim.synchrotron import *

def MaxwellBoltzmann(spectrum, T, n):
    # TODO: Implement the Maxwell-Boltzmann distribution for the particle spectrum
    spectrum.spec = np.exp(-spectrum.E / T)
    spectrum.set_norm(n)
    return

def Powerlaw(spectrum, alpha, Emin, Emax, n):
    # TODO: Implement the power-law distribution for the particle spectrum
    spectrum.spec = np.where((spectrum.E >= Emin) & (spectrum.E <= Emax), spectrum.E ** (-alpha), 0)
    spectrum.set_norm(n)
    return

if __name__ == "__main__":
    T = 1e5
    grid = Grid(dimension=3, dx=0.1, n_mesh=21, data_type=float)
    particle = Particle(grid = grid,
                        mass=1,
                        charge=-1,
                        gm_min=1e-1,
                        gm_max=1e10,
                        n_spectrum=101)
    
    for idx in np.ndindex(particle.value.shape):
        MaxwellBoltzmann(particle.value[idx], T, 1e5)

    print("Particle values initialized.")

    photon = Photon(grid = grid,
                    Emin=1e-1,
                    Emax=1e10,
                    n_spectrum=101)
    

    B = 1e-6
    # prepare Function Table
    Fx = F()
    for idx in np.ndindex(particle.value.shape):
        synchrotron(photon.value[idx], particle.value[idx], B, Fx)
        print(idx)
    
    print("Photon values initialized.")

    # particle[0,0,0].plot(loglog=True, gm_compensate=2.0)
    photon[0,0,0].plot(loglog=True, power_compensate=2.0)
    plt.show()
    
