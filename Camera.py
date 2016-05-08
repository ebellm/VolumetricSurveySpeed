"""class for calculating performance of straw-man optical cameras."""

from Efficiency import Efficiency
import numpy as N
import os
import inspect

# directory where the reduction code is stored
BASE_DIR = os.path.dirname(os.path.abspath(inspect.getfile(
    inspect.currentframe()))) +'/'

class Camera:
    def __init__(self, DetectorType, num_detectors,beam_obstruction=0.):
        self.Detector = DetectorType
        self.num_detectors = num_detectors
        self.beam_obstruction = beam_obstruction # float < 1
        self.filters = {}

    def active_area(self):
        return self.num_detectors * self.Detector.active_area()

    def add_filter(self,Filter):
        self.filters[Filter.name] = Filter


class Detector:
    def __init__(self, x_dim, y_dim, npix_x, npix_y, QE, readnoise, full_well,
        dark_current,ADU_bits=16,gain=None):
        
        self.x_dim = x_dim # in cm
        self.y_dim = y_dim # in cm
        self.npix_x = npix_x
        self.npix_y = npix_y
        self.QE = QE
        self.readnoise = readnoise # in electrons
        self.full_well = full_well # e-
        self.ADU_bits = ADU_bits
        self.max_ADU = 2**ADU_bits -1
        if gain is None:
            self.gain = self.full_well/self.max_ADU # e-/ADU
        else:
            self.gain = gain
        self.dark_current = dark_current # e-/s/pix
        # dark current varies as T^1.5 exp (Eg(T)/kT)
        # readout time?

    def active_area(self):
        return self.x_dim * self.y_dim
        
    def pixel_pitch(self):
        # cm/pixel
        # average, since e2v are slightly asymmetric
        return N.mean([self.x_dim/self.npix_x,self.y_dim/self.npix_y])


########## QE CURVES ###############

# for simplicity, ignore the high resistivity chips
PTF_QE = Efficiency(name='CFHQE_EPI')
PTF_QE.from_file(BASE_DIR+'data/CFH_QE_EPI.csv',wavelength_unit='nm',
    efficiency_rescale=0.01, delimiter=',')

e2v_QE_ZTF= Efficiency(name='e2vQE_ZTF')
e2v_QE_ZTF.from_file(BASE_DIR+'data/e2v_intermediate_email130326.csv',
    wavelength_unit='nm',efficiency_rescale=1.0,delimiter=',')

e2v_QE_multi2= Efficiency(name='e2vQE_multi2')
e2v_QE_multi2.from_file(BASE_DIR+'data/e2v_multi2_email130118.csv',
    wavelength_unit='nm',efficiency_rescale=1.0,delimiter=',')

DECam_QE = Efficiency(name='DECam_QE')
DECam_QE.from_file(BASE_DIR+'data/DES_CCD_QE.txt')

LSST_QE = Efficiency(name='LSSTQE')
LSST_QE.from_file(BASE_DIR+'data/LSST_throughputs-1.2/baseline/detector.dat',
    wavelength_unit='nm')

mirror_reflectivity = Efficiency(name='mirror')
mirror_reflectivity.from_file(
    BASE_DIR+'data/mirror_reflectivity_aluminum_Gemini.csv',
    wavelength_unit='nm',delimiter=',')

########## FILTER TRANSMISSIONS ###############

# check--newer filter curves in DES_filter_curves.txt
DESg = Efficiency(name='DESg')
DESg.from_file(BASE_DIR+'data/DESg.csv',
    wavelength_unit='nm',efficiency_rescale=0.01,delimiter=',')
DESr = Efficiency(name='DESr')
DESr.from_file(BASE_DIR+'data/DESr.csv',
    wavelength_unit='nm',efficiency_rescale=0.01,delimiter=',')

PS1g = Efficiency(name='PS1g')
PS1g.from_file(BASE_DIR+'data/PS1g.csv',
    wavelength_unit='nm',delimiter=',')
PS1r = Efficiency(name='PS1r')
PS1r.from_file(BASE_DIR+'data/PS1r.csv',
    wavelength_unit='nm',delimiter=',')

uprime = Efficiency(name='uprime')
uprime.from_file(BASE_DIR+'data/sdss_uprime_transmission_flwo.txt',
    wavelength_unit='nm',efficiency_rescale=0.01)
gprime = Efficiency(name='gprime')
gprime.from_file(BASE_DIR+'data/sdss_gprime_transmission_flwo.txt',
    wavelength_unit='nm',efficiency_rescale=0.01)
rprime = Efficiency(name='rprime')
rprime.from_file(BASE_DIR+'data/sdss_rprime_transmission_flwo.txt',
    wavelength_unit='nm',efficiency_rescale=0.01)
iprime = Efficiency(name='iprime')
iprime.from_file(BASE_DIR+'data/sdss_iprime_transmission_flwo.txt',
    wavelength_unit='nm',efficiency_rescale=0.01)
zprime = Efficiency(name='zprime')
zprime.from_file(BASE_DIR+'data/sdss_zprime_transmission_flwo.txt',
    wavelength_unit='nm',efficiency_rescale=0.01)

MouldR = Efficiency(name='MouldR')
MouldR.from_file(BASE_DIR+'data/Mould-R_cfh7603.dat',
    wavelength_unit='nm',efficiency_rescale=0.01)
MouldR.from_arrays(MouldR.wavelength[::-1],MouldR.efficiency[::-1])

LSSTu = Efficiency(name='LSSTu')
LSSTu.from_file(BASE_DIR+'data/LSST_throughputs-1.2/baseline/filter_u.dat',
    wavelength_unit='nm')
LSSTg = Efficiency(name='LSSTg')
LSSTg.from_file(BASE_DIR+'data/LSST_throughputs-1.2/baseline/filter_g.dat',
    wavelength_unit='nm')
LSSTr = Efficiency(name='LSSTr')
LSSTr.from_file(BASE_DIR+'data/LSST_throughputs-1.2/baseline/filter_r.dat',
    wavelength_unit='nm')
LSSTi = Efficiency(name='LSSTi')
LSSTi.from_file(BASE_DIR+'data/LSST_throughputs-1.2/baseline/filter_i.dat',
    wavelength_unit='nm')
LSSTz = Efficiency(name='LSSTz')
LSSTz.from_file(BASE_DIR+'data/LSST_throughputs-1.2/baseline/filter_z.dat',
    wavelength_unit='nm')
LSSTy = Efficiency(name='LSSTy')
LSSTy.from_file(BASE_DIR+'data/LSST_throughputs-1.2/baseline/filter_y.dat',
    wavelength_unit='nm')

########## DETECTORS ###############

e2v_ZTF = Detector(9.20,9.20,6144,6144,e2v_QE_ZTF,11,2.0E5,(20./3600.),
    gain=1.13)

PTF_det = Detector(3.,6.,2e3,4e3,PTF_QE,12,6e4,(1./60.),gain=1.6)

# DECam
DECam_det = Detector(6.144,3.072, 4096, 2048, DECam_QE, 7, 1e4, 25./3600.,
    ADU_bits=16, gain=4.5)

# LSST
# readnoise is supposed to be < 5 electrons
# size is a straight multiplication of 10 micron pixels
LSST_det = Detector(4.096,4.096,4096,4096,LSST_QE,5,
# full well, gain are all stolen from DECam for now...
# dark current of 15 e-/s/pix from LSST ETC
    1e4, 15, ADU_bits=16, gain=4.5)
    

########## CAMERAS ###############

PTF_cam = Camera(PTF_det,11,beam_obstruction=0.15)
PTF_cam.add_filter(gprime)
PTF_cam.add_filter(MouldR)

ZTF_cam = Camera(e2v_ZTF,16,beam_obstruction=0.21)
ZTF_cam.add_filter(gprime)
ZTF_cam.add_filter(iprime)
ZTF_cam.add_filter(MouldR)

DECam = Camera(DECam_det,62)
DECam.add_filter(DESr)
DECam.add_filter(DESg)

LSST_cam = Camera(LSST_det,21*9)
LSST_cam.add_filter(LSSTu)
LSST_cam.add_filter(LSSTg)
LSST_cam.add_filter(LSSTr)
LSST_cam.add_filter(LSSTi)
LSST_cam.add_filter(LSSTz)
LSST_cam.add_filter(LSSTy)
