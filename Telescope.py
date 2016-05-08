
from Efficiency import Efficiency
from Magnitudes import ABmag2flux
from Camera import PTF_cam, ZTF_cam, LFC_cam, WaSP_cam
from Camera import WIYN_cam, DECam, ATLAS_cam, LSST_cam
from Camera import BlackGEM_cam
import Atmosphere
import numpy as N
from scipy import integrate

IN2CM = 2.54
PLANCKH = 6.6261E-27 #erg s
NM2CM = 1E-7

class Telescope:
    def __init__(self, lat, lon, alt, name, diameter, plate_scale, transmission):
        self.lon = lon # (unused)
        self.lat = lat # (unused)
        self.elevation = alt # meters
        self.name = name # (unused)
        self.diameter = diameter #cm
        self.plate_scale = plate_scale # arcsec/mm
        # telescope transmission
        self.transmission = transmission # a function of wavelength: specify
            # a scalar, a dictionary by filterkey, (or TODO: an Efficiency) 

    def set_camera(self, camera):
        self.Camera = camera
        self.pixel_scale = self.Camera.Detector.pixel_pitch()* 10. * \
            self.plate_scale # arcsec/pixel

    def npix(self, seeing_fwhm, aperture='aper'):
        """number of pixels in aperture extraction region.
        seeing in arcsec
        
        if aperture == 'aper', use the SNR-maximizing extraction radius
            (1.346 FWHM)--see LSST SNR doc eq. 19
        if aperture == 'sex', use the number of pixels used by Sextractor
            (2.5 Kron radius)--see 5/22/13 notes.
        if aperture == 'eran', use the 1.25*fwhm radius used in the IPAC
            PTF limiting mag calcuation"""
        # LSST SNR doc eqn 19

        assert (aperture in ('aper','eran','sex'))

        # for undersampled data this is probably an underestimate of the noise
        if aperture == 'aper':
            npix_extract = N.pi * (0.673 * seeing_fwhm / self.pixel_scale)**2.
        elif aperture == 'sex':
            npix_extract = N.pi * (1.33 * seeing_fwhm / self.pixel_scale)**2.
        elif aperture == 'eran':
            npix_extract = N.pi * (1.25 * seeing_fwhm / self.pixel_scale)**2.

        # don't return fractional pixels
        npix_extract = N.round(npix_extract)

        if npix_extract < 1.:
            return 1.
        else:
            return npix_extract

    def telescope_transmission(self,filterkey):
        
        if type(self.transmission) == float:
            return self.transmission * (1. - self.Camera.beam_obstruction)
        elif type(self.transmission) == dict:
            try:
                return self.transmission[filterkey] * \
                    (1. - self.Camera.beam_obstruction)
            except KeyError:
                raise NotImplementedError('Transmission not specified for given filter')
        else:
            raise NotImplementedError('Invalid transmission format')


    def atm_transmission(self,airmass):
        return Atmosphere.atm_transmission(airmass,altitude=self.elevation)

    def ADU_per_electron(self):
        return 1./self.Camera.Detector.gain

    def AB_to_Rstar(self,source_mag,filterkey='gprime',airmass=1.,
        aperture_cut=True, absorb=True, aperture='aper'):
        # returns electrons per second
        # form taken from LSST S/N doc equations 3-4, Mike Bolte SNR notes

        assert (aperture in ('aper','eran','sex'))

        fnu = ABmag2flux(source_mag,cgs=True) # scalar value for now
        if absorb:
            atm = self.atm_transmission(airmass)
        else:
            # don't absorb the sky background
            atm = 1.
        const = N.pi * self.diameter**2. / \
            (4. * PLANCKH)
        # multiply Efficiencies to get them on the same grid
        integrand = const * fnu * self.telescope_transmission(filterkey) \
            * atm * (self.Camera.filters[filterkey] * \
            self.Camera.Detector.QE)
        # divide by wavelength (energy flux to photon flux)
        integrand_array = integrand.efficiency/(integrand.wavelength * NM2CM)
        rstar = integrate.trapz(integrand_array,
            x=(integrand.wavelength * NM2CM))
        
        # scale to aperture electrons at 1.36 FWHM: 71.5%  (LSST S/N doc eqn 18)
        if aperture_cut:
            if aperture == 'aper':
                return rstar * (1. - N.exp(-0.5 * 1.585**2.))
            elif aperture == 'sex':
                return rstar * (1. - N.exp(-0.5 * 3.133**2.))
            elif aperture == 'eran':
                return rstar * (1. - N.exp(-0.5 * 2.944**2.))
        else:
            return rstar

    def Rstar_to_AB(self,Rstar,filterkey='gprime',airmass=1.,aperture='aper'):
        Rstar20 = self.AB_to_Rstar(20.,filterkey=filterkey,airmass=airmass,
            aperture=aperture)
        return 20. - 2.5*N.log10(Rstar/Rstar20)

    def sky_electrons_per_pixel(self,mag_per_sq_arcsec,filterkey='gprime'):
        # returns electrons per pixel per second
        # area of one pixel in arcsec^2.
        pixarea = (self.plate_scale*self.Camera.Detector.pixel_pitch()*10.)**2.
        mag_per_pix = mag_per_sq_arcsec - 2.5 * N.log10(pixarea)
        # could store R20 = R(20) and do R(m) = R(20) * 10**(0.4 * (20-m))
        return self.AB_to_Rstar(mag_per_pix,filterkey=filterkey,airmass=1.,
            aperture_cut=False,absorb=False)
    
    def integration_time(self,source_mag,seeing_fwhm,sky_brightness,
        filterkey='gprime',airmass=1.,SNR=5,aperture='aper'):
        
        npix = self.npix(seeing_fwhm, aperture=aperture)
        Rstar = self.AB_to_Rstar(source_mag,filterkey=filterkey,airmass=airmass,
            aperture=aperture)
            
        Rsky = self.sky_electrons_per_pixel(sky_brightness,filterkey=filterkey)
        # TODO doesn't include dark current or digitization noise on RN
        return ( (SNR**2. * (Rstar + npix*Rsky)) + N.sqrt( SNR**4. * (Rstar + 
            npix*Rsky)**2 + 4. * (Rstar*SNR*self.Camera.Detector.readnoise)**2.
            * npix) )/(2*Rstar**2.)
        # sky limited case:
        #return SNR**2.*npix*Rsky/Rstar**2.
        
    def limiting_mag(self,time,seeing_fwhm,sky_brightness,
            filterkey='gprime',airmass=1.,SNR=5.,aperture='aper'):
        npix = self.npix(seeing_fwhm, aperture=aperture)
        Rsky = self.sky_electrons_per_pixel(sky_brightness,filterkey=filterkey)

        # looks like X is just total background (in electrons)
        X = Rsky*time*npix + (self.Camera.Detector.readnoise**2. + \
            (self.Camera.Detector.gain/2.)**2.)*npix + \
            self.Camera.Detector.dark_current*npix*time
        Rstar = (SNR**2. * time + N.sqrt((SNR**2.*time)**2. + \
            4. * time**2. * SNR**2. * X))/(2*time**2.)
        ## sky limited case:
        #Rstar = N.sqrt(SNR**2.*npix*Rsky/time)
        return self.Rstar_to_AB(Rstar,filterkey=filterkey,airmass=airmass,
            aperture=aperture)

    def signal(self,source_mag,time,filterkey='gprime',airmass=1.,
        aperture='aper'):
        # returns electrons

        Rstar = self.AB_to_Rstar(source_mag,filterkey=filterkey,airmass=airmass,
                aperture_cut=True,aperture=aperture)
        return  Rstar * time 

    def noise(self,source_mag,time,seeing_fwhm,sky_brightness, 
            filterkey='gprime',airmass=1.,aperture='aper'):
        # returns electrons

        npix = self.npix(seeing_fwhm, aperture=aperture)
        Rsky = self.sky_electrons_per_pixel(sky_brightness,filterkey=filterkey)
        Rstar = self.AB_to_Rstar(source_mag,filterkey=filterkey,airmass=airmass,
                aperture_cut=True,aperture=aperture)
        
        # all of these are squared
        shot_noise2 = Rstar * time
        sky_noise2 = Rsky * time * npix
        # includes digitization noise
        read_noise2 = (self.Camera.Detector.readnoise**2. + \
            (self.Camera.Detector.gain/2.)**2.)*npix 
        dark_noise2 = self.Camera.Detector.dark_current*npix*time 

        return  N.sqrt(shot_noise2 + sky_noise2 + read_noise2 + dark_noise2)


    def SNR(self,source_mag,time,seeing_fwhm,sky_brightness, 
            filterkey='gprime',airmass=1.,aperture='aper'):
        return self.signal(source_mag,time,filterkey=filterkey,airmass=airmass,
            aperture=aperture) / self.noise(source_mag,time,seeing_fwhm,
            sky_brightness, filterkey=filterkey,airmass=airmass,
            aperture=aperture)



# coords from Harrington PASP 1952
# transmission for the telescope alone a guess from Law's FRD, dividing off
# rough filter transmission
# additional factor of 1.1 used to scale to peak of observed R-band data
# (see notes, 7/15/2012, 6/12/2013)
# Note that I need a transmission > 1 to match the database g' limits! (5/12/13)
# set iprime=MouldR
P48 = Telescope('33:21:26.35','-116:51:32.04',1707,'P48',48*IN2CM,67.187,
    {'MouldR':0.64/0.9*1.1,'gprime':0.64/0.9*1.568,'iprime':0.64/0.9*1.1})
P48.set_camera(PTF_cam)

# mirror diameter is effective based on 10.0 m^2 light collecting area quoted at http://www.ctio.noao.edu/noao/content/basic-optical-parameters
# telescope transmission is a rough DESr-band average of the CFHT/MEGACam primary/corrector transmission files given in DESr-FullSize-woptics-watmos.par
Blanco = Telescope('-30:10:10.78','-70:48:23.49',2215,'Blanco',178.4*2.,18,0.74)
Blanco.set_camera(DECam)

# LSST effective aperture
# transmission is goosed to match the value in the LSST SRD Table 6:
#  2x15 sec, 0.7", 21 mag/sq arcsec, r = 24.7 (PSF extraction)
# per the LSST SNR doc, that's 24.6 in aperture 
LSST = Telescope('-30:14:40.7','-70:44:57.9',2662.75,'LSST',668,19.64,0.9)
LSST.set_camera(LSST_cam)
