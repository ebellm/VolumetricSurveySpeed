from __future__ import division
import numpy as N
from Telescope import P48, Blanco
from Camera import PTF_cam, ZTF_cam
import cosmolopy as cp
import matplotlib.pyplot as plt
from scipy.integrate import quad
from scipy.optimize import golden

SR_TO_SQ_DEG = 3282.8
SIDEREAL_DAY_SEC = 23.9344699*3600.

SEC2HR = 1./3600.
HR2SEC = 1./SEC2HR
MIN2HR = 1./60.
DAY2HR = 24.
HR2DAY = 1./DAY2HR
SEC2DAY= SEC2HR*HR2DAY
DAY2SEC = DAY2HR * HR2SEC
LUN2HR = 24*28.
YEAR2HR = 365.25*24.

def limiting_z(apparent_mag, absolute_mag, k_corr = None):
    # solve for redshift of source given its apparent & absolute mags
    # use k-correction for an f_lambda standard: k = -2.5 log10(1./(1+z))
    # see Hogg 99 eqn. 27

    if k_corr is None:
        k_corr = lambda z: -2.5 * N.log10(1./(1.+z))

    def f(z): 
        if z > 0:
            # abs to use minimization routines rather than root finding
            return N.abs(absolute_mag + \
                cp.magnitudes.distance_modulus(z,**cp.fidcosmo) + k_corr(z) - \
                apparent_mag)
        else:
            # don't let it pass negative values
            return N.inf

    #res = brute(f, ((1e-8,10),), finish=fmin, full_output=True)
    res = golden(f)
    return res



def volumetric_survey_rate(absolute_mag,
        snapshot_area_sqdeg, DIQ_fwhm_arcsec,slew_time=15.,label=None,
        sky_brightness=None,transmission=None,plot=True,readnoise=None,
        telescope = P48, camera = ZTF_cam, filterkey='MouldR', 
        max_lim_mag = None, obstimes = None, k_corr=None, **kwargs):
    """calculate the volume/sec/snapshot in Mpc^3"""

    if obstimes is None:
        obstimes = N.logspace(0,2,100) # seconds
        #obstimes = N.linspace(5,100,20) # seconds
        #obstimes = N.array([30,45,60,120,180,300,500])
    if transmission is not None:
        raise NotImplementedError('check for correctness: varying camera obscuration now incorported in Camera.beam_obscuration')
        telescope.transmission = transmission
    if camera is not None:
        telescope.set_camera(camera)
    if readnoise is not None:
        telescope.Camera.Detector.readnoise = readnoise
    if sky_brightness is None:
        # half moon in both g' and r'
        sky_brightness = 19.9
    limiting_mags = N.array([telescope.limiting_mag(time,DIQ_fwhm_arcsec,
        sky_brightness, airmass=1.15,filterkey=filterkey) for time in obstimes])

    if max_lim_mag is not None:
        limiting_mags[limiting_mags >= max_lim_mag] = max_lim_mag

    exptimes = obstimes + slew_time

    zs = [limiting_z(m, absolute_mag, k_corr=k_corr) for m in limiting_mags]
    com_volumes = cp.distance.comoving_volume(zs,**cp.fidcosmo)


    vol_survey_rate = com_volumes * \
        (snapshot_area_sqdeg / (4. * N.pi * SR_TO_SQ_DEG)) / exptimes

    if plot:
        plt.plot(obstimes,vol_survey_rate,label=label,**kwargs)
        plt.xlabel('Integration time (sec)')
        plt.ylabel('Volumetric Survey Rate per Exposure (Mpc$^3$ s$^{-1}$)')
        plt.xscale('log')
        plt.yscale('log')
        plt.xlim([obstimes.min(),obstimes.max()])
    if False:
        plt.plot(obstimes,limiting_mags,label=label)
        plt.xlabel('Integration time (sec)')
        plt.ylabel('Limiting Magnitude ({})'.format(filterkey))
        plt.xscale('log')
        plt.yscale('linear')
        plt.xlim([obstimes.min(),obstimes.max()])

    #print vol_survey_rate.max(), obstimes[vol_survey_rate.argmax()]
    return vol_survey_rate.max(), obstimes[vol_survey_rate.argmax()]
    #return vol_survey_rate, obstimes, limiting_mags

def spectroscopic_cost(z,absolute_mag):
    """defines a cost (in terms of fractions of a night) needed for followup
    classification spectroscopy.  Numbers are rough, but scale for P200:
    20 minutes for a mag 20 target, plus 5 minutes of overhead independent
    of the magnitude.  Normalize by a 6-hour night (average, with Palomar 
    weather)"""
    
    DM = cp.magnitudes.distance_modulus(z,**cp.fidcosmo)
    mag = DM + absolute_mag
    # background-limited exposure time at constant S/N
    # t_exp[mag_ref] * 10**(0.8(mag - mag_ref)) + t_OH
    mag_ref = 20
    texp_ref = 20. # minutes
    t_oh = 5. # minutes, overhead/minimum exposure time
    return (texp_ref * 10**(0.8*(mag - mag_ref)) + t_oh)/360.

def unweighted_survey_volume(absolute_mag,limiting_mag, k_corr = None):
    """determine what spatial volume a survey can see absolute_mag objects to
    given its limiting_mag.  """

    # TODO add saturation magnitude for a lower limit...
    # or just call with limiting_mag = saturation mag and subtract

    z_limit = limiting_z(limiting_mag, absolute_mag, k_corr = k_corr)

    # testing the integration
    return cp.distance.comoving_volume(z_limit,**cp.fidcosmo)

def unweighted_survey_speed(absolute_mag,limiting_mag,fov,time_per_obs,
    k_corr = None):
    return (unweighted_survey_volume(absolute_mag,limiting_mag,
        k_corr = k_corr) * 
        (fov / (4. * N.pi * SR_TO_SQ_DEG)) / time_per_obs)

def fraction_spectroscopic_volume(absolute_mag,limiting_mag,
    spectroscopic_limit=21, k_corr = None):
    frac =  (unweighted_survey_volume(absolute_mag,spectroscopic_limit,
        k_corr = k_corr)/  
        unweighted_survey_volume(absolute_mag,limiting_mag, k_corr=k_corr)) 
    if frac > 1.:
        return 1
    else:
        return frac

def weighted_survey_volume(absolute_mag,limiting_mag, k_corr = None):
    """determine what spatial volume a survey can see absolute_mag objects to
    given its limiting_mag.  Weight the volume elements by the cost of
    spectroscopic followup (in fraction of a night) at that distance."""

    # TODO add saturation magnitude for a lower limit...

    z_limit = limiting_z(limiting_mag, absolute_mag, k_corr = k_corr)

    # testing the integration
    #print cp.distance.comoving_volume(z_limit,**cp.fidcosmo)
    #print 4*N.pi*quad(lambda z: cp.distance.diff_comoving_volume(z,
    #        **cp.fidcosmo), 0,z_limit)[0]

    # integrate the cost function over the volume
    return 4*N.pi*quad(lambda z: cp.distance.diff_comoving_volume(z,
            **cp.fidcosmo)/spectroscopic_cost(z,absolute_mag), 0,z_limit)[0]
    

def compare_weighted_survey_speed(absolute_mag,limiting_mag,fov,time_per_obs):
    """weighted survey speed relative to PTF"""
    
    ptf = (weighted_survey_volume(absolute_mag,20.7) * 
        (7.26 / (4. * N.pi * SR_TO_SQ_DEG)) / 106.)

    other = (weighted_survey_volume(absolute_mag,limiting_mag) * 
        (fov / (4. * N.pi * SR_TO_SQ_DEG)) / time_per_obs)

    return other/ptf
        
def wrap_survey_speeds(absolute_mag,fov,time_per_obs,limiting_mag,
    ptfspeed=2790.):
    # use for survey comparison tables

    # ptf speed = unweighted_survey_speed(-19,21,7.26,106) * 1.0 = 4001
    # ptf speed = unweighted_survey_speed(-19,20.7,7.26,106) * 1.0 = 2790
    # ptf speed = unweighted_survey_speed(-19,20.6,7.26,106) * 1.0 = 2473

    speed = unweighted_survey_speed(absolute_mag,limiting_mag,fov,time_per_obs)
    frac = fraction_spectroscopic_volume(absolute_mag,limiting_mag)
    omega_dot = float(fov)/time_per_obs * 3600.
    nexps = n_exposures_per_field_per_year(camera_fov_sqdeg=fov,
        time_per_image_sec=time_per_obs)
    vdot = speed
    fspec = frac

    print "{:d} & {:d} & \\num{{ {:.1e} }} & {:.2f} \\\\".format(
                int(omega_dot), int(nexps), vdot, fspec)

def linear_control_time(absolute_mag, limiting_mag, tau_eff, z, k_corr = None):
    # estimate of control time from simple light curve model: see 8/18/15 notes
    # absolute_mag is peak absolute mag
    # tau_eff is days to decline 1 mag
    # returns control time in days

    DM = cp.magnitudes.distance_modulus(z,**cp.fidcosmo)

    # use k-correction for an f_lambda standard: k = -2.5 log10(1./(1+z))
    # see Hogg 99 eqn. 27

    if k_corr is None:
        k_corr = lambda z: -2.5 * N.log10(1./(1.+z))

    apparent_mag_peak = absolute_mag + DM + k_corr(z)

    ct = -1.*(apparent_mag_peak - limiting_mag) * tau_eff * (1. + z)

    if ct > 0:
        return ct
    else:
        return 0.

def sum_control_time_one_consecutive(obs_points_jd, control_time_days):
    # control time is in observer frame
    # returns years

    dt_obs_days = N.diff(obs_points_jd)

    # see Zwicky 1942
    w_dt_lt = dt_obs_days < control_time_days

    return (control_time_days + N.sum(dt_obs_days[w_dt_lt]) + \
        N.sum(~w_dt_lt)*control_time_days)/365.25

def sum_control_time_k_consecutive(obs_points_jd, control_time_days, 
    k_consecutive):
    # control time is in observer frame
    # returns years
    
    dt_obs_days = obs_points_jd[(k_consecutive-1):] - \
        obs_points_jd[:-(k_consecutive-1)]

    ctj_prev = 0.
    ct_sum = 0.
    for dtj in dt_obs_days:
        if dtj > control_time_days:
            ctj = 0.
        else:
            if ctj_prev == 0.:
                ctj = control_time_days
            else:
                ctj = dtj
        ct_sum += ctj
        ctj_prev = ctj

    return ct_sum/365.25
        
def sum_control_time(obs_points_jd, control_time_days, k_consecutive=1):
    # control time is in observer frame
    # returns years

    if k_consecutive == 1:
        return sum_control_time_one_consecutive(obs_points_jd, 
            control_time_days)
    else:
        return sum_control_time_k_consecutive(obs_points_jd, 
            control_time_days, k_consecutive)

def n_transients_per_year(survey, absolute_mags, tau_effs, cadence_days, 
    rate_z = None, k_corr = None, max_mlim = None,
        max_zenith_angle=66.4, k_consecutive=1):
    # rate a function of z in events Mpc^-3 yr^-1
    # for efficiency, require scalar cadences

    obs_points_jd = survey.yearly_cadence_points(cadence_days)

    if rate_z is None:
        # use the Ia rate by default (see LSST SB)
        rate_z = lambda z: 3.E-5 

    doy = N.arange(365)

    n_events = N.zeros([absolute_mags.size,tau_effs.size])

    for i, absolute_mag in enumerate(absolute_mags):
        # we're being a little fast and loose here with the snapshot/all-sky distinction
        area, vol, mlim = zip(*[survey.snapshot_size(cadence_days, doy=d, max_zenith_angle=max_zenith_angle, absolute_mag = absolute_mag, max_mlim = max_mlim, k_corr=k_corr) for d in doy])

        snap_area = N.mean(area)
        limiting_mag = N.mean(mlim)

        z_limit = limiting_z(limiting_mag, absolute_mag, k_corr = k_corr)

        for j, tau_eff in enumerate(tau_effs):
            def integrand(z):
                ctz = linear_control_time(absolute_mag, limiting_mag, 
                    tau_eff, z, k_corr = k_corr)
                return rate_z(z) / (1. + z) * \
                    sum_control_time(obs_points_jd,ctz, \
                    k_consecutive = k_consecutive) * \
                    snap_area/SR_TO_SQ_DEG * \
                    cp.distance.diff_comoving_volume(z,**cp.fidcosmo)

            n_events[i,j] = quad(integrand,0,z_limit)[0]
            print absolute_mag, tau_eff, n_events[i,j]

    return n_events
    
