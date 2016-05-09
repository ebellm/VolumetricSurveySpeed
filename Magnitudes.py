
import numpy as N

JY2CGS = 1E-23  # erg s^-1 cm^-2 Hz^-1


def ABmag2flux(magnitude, cgs=False):
    if cgs:
        return 3631 * 10.**(-0.4 * magnitude) * JY2CGS
    else:
        return 3631 * 10.**(-0.4 * magnitude)


def flux2ABmag(flux, cgs=False):
    if cgs:
        return -2.5 * N.log10(flux) - 48.60
    else:
        return -2.5 * N.log10(flux * JY2CGS) - 48.60
