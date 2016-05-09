import numpy as N
import scipy.interpolate as interp
import matplotlib.pyplot as plt
import types
from scipy import integrate
NM2ANG = 10.
C_NMS = 2.99792E17  # speed of light, nanometers per second (!)
H_CGS = 6.626E-27  # erg/s


class Efficiency:
    """array of efficiencies by wavelength (0-1)"""

    def __init__(self, name=None):
        self.name = name

    def from_file(self, filename, wavelength_unit='angstrom',
                  efficiency_rescale=1., delimiter=None, **kwargs):
        # file is a simple 2d text file with wavelength and value
        arr = N.genfromtxt(filename, comments='#', delimiter=delimiter,
                           names=['wavelength', 'efficiency'], **kwargs)

        # store in nm
        if wavelength_unit == 'angstrom':
            arr['wavelength'] /= NM2ANG
        elif wavelength_unit == 'nm':
            pass
        else:
            raise ValueError('invalid wavelength units')

        self.wavelength = arr['wavelength']
        self.efficiency = arr['efficiency'] * efficiency_rescale
        self.frequency = C_NMS / self.wavelength

    def to_file(self, filename):
        N.savetxt(filename, N.array((self.wavelength, self.efficiency)).T)

    def from_arrays(self, wavelength, efficiency):
        self.wavelength = N.array(wavelength)
        self.efficiency = N.array(efficiency)
        self.frequency = C_NMS / self.wavelength

    def __mul__(self, E2):
        """interpolate efficiencies and multiply"""
        if isinstance(E2, (types.FloatType, types.IntType)):
            Eout = Efficiency(name=self.name + 'x')
            Eout.from_arrays(self.wavelength, self.efficiency * E2)
            return Eout
        w1 = self.wavelength
        w2 = E2.wavelength
        # if the wavelength grids are identical this is easy
        if N.all(w1 == w2):
            Eout = Efficiency(name=self.name + 'x' + E2.name)
            Eout.from_arrays(w1, self.efficiency * E2.efficiency)
            return Eout
        # otherwise we have to interpolate
        med_w1 = N.median(w1[1:] - w1[:-1])
        med_w2 = N.median(w2[1:] - w2[:-1])
        # interpolate to the finer pitch
        if med_w1 < med_w2:
            interp_from = E2
            interp_to = self
        else:
            interp_from = self
            interp_to = E2
        f = interp.interp1d(interp_from.wavelength, interp_from.efficiency,
                            bounds_error=False, fill_value=0.)
        vals = f(interp_to.wavelength) * interp_to.efficiency
        Eout = Efficiency(name=self.name + 'x' + E2.name)
        Eout.from_arrays(interp_to.wavelength, vals)
        return Eout

    def __rmul__(self, E2):
        return self.__mul__(E2)

    def __pow__(self, power):
        Eout = Efficiency(name=self.name + '^' + str(power))
        Eout.from_arrays(self.wavelength, self.efficiency**power)
        return Eout

    def __div__(self, E2):
        return self.__mul__(E2**(-1.))

    def __rdiv__(self, E2):
        Einv = self**(-1.)
        return E2 * Einv

    def integrate(self):
        # TODO: integral bounds
        return integrate.trapz(self.efficiency, x=self.wavelength)

    def integrate_dlognu(self):
        # TODO: integral bounds
        # AB mag integrals are d(log nu) = 1/(h nu) d nu
        return integrate.trapz(self.efficiency / (H_CGS * self.frequency),
                               x=self.frequency)

    def range_above(self, emin):
        """return (interpolated) wavelength range where efficiency > emin"""

        w = N.flatnonzero(self.efficiency >= emin)

        if (w[0] == 0) or (w[-1] == len(self.efficiency) - 1):
            raise ValueError("range not bounded")

        fmin = interp.interp1d(self.efficiency[w[0] - 1:w[0] + 1],
                               self.wavelength[w[0] - 1:w[0] + 1],
                               bounds_error=True, fill_value=0.)

        wmin = fmin(emin)

        fmax = interp.interp1d(self.efficiency[w[-1]:w[-1] + 2],
                               self.wavelength[w[-1] - 1:w[-1] + 1],
                               bounds_error=True, fill_value=0.)

        wmax = fmax(emin)

        return (wmin.tolist(), wmax.tolist())

    def weighted_average_wavelength(self, wave_range=None):

        if wave_range is not None:
            w = ((self.wavelength >= wave_range[0]) &
                 (self.wavelength <= wave_range[1]))
            enew = Efficiency('enew')
            enew.from_arrays(self.wavelength[w], self.efficiency[w])
        else:
            enew = self

        elambda = Efficiency('lambda')
        elambda.from_arrays(enew.wavelength, enew.wavelength)

        return (enew * elambda).integrate() / enew.integrate()

    def plot(self, label=None, **kwargs):
        if label is None:
            label = self.name
        plt.plot(self.wavelength, self.efficiency, label=label, **kwargs)
        plt.xlabel('Wavelength (nm)')
        plt.ylabel('Efficiency')
