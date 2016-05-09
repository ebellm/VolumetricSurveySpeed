"""atmospheric absorption"""
from Efficiency import Efficiency
import numpy as N
import os
import inspect

# directory where the reduction code is stored
BASE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))) + '/'

atm_depth = Efficiency(name='depth')
# convert from magnitudes extinction to zenith optical depth
# see Dec. 12 2011 notes, or
# http://ganymede.nmsu.edu/holtz/a535/ay535notes/node3.html
atm_depth.from_file(BASE_DIR + 'data/palomar_extinction.txt',
                    wavelength_unit='angstrom', efficiency_rescale=(-1. / 1.086))
# transmission is then exp(-atm_depth * airmass)


def atm_transmission(airmass, altitude=1700.):
    trans = N.exp(atm_depth.efficiency * airmass *
                  N.exp((1700 - altitude) / 7000))
    Trans = Efficiency(name='transmission')
    Trans.from_arrays(atm_depth.wavelength, trans)
    return Trans
