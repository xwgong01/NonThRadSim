import numpy as np 
import matplotlib.pyplot as plt

class Spectrum():

    def __init__(self, Emin, Emax, shape, dtype = np.float32):
        self.Emin = Emin
        self.Emax = Emax
        self.shape = shape
        self.E = np.geomspace(self.Emin, self.Emax, self.shape)
        self.spec = np.zeros(self.shape, dtype=dtype)
        self.norm = None
        
    def get_norm(self):
        return self.norm
    
    def set_norm(self, norm):
        self.norm = norm
        if self.spec.sum() <=0:
            raise ValueError("Spectrum values must be positive to normalize.")
        normcoeff = self.norm / np.trapezoid(self.spec, self.E)
        self.spec *= normcoeff
        return 
    
    def plot(self, ax=None, loglog=True, power_compensate = 0.0,**kwargs):
        compensate = self.E ** power_compensate
        if ax is None:
            fig, ax = plt.subplots()
        if loglog:
            ax.loglog(self.E, compensate*self.spec, **kwargs)
        else:
            ax.plot(self.E, compensate*self.spec, **kwargs)
        return ax
