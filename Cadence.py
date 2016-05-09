
import numpy as N
import pandas
from scipy.optimize import brentq
import astropy.coordinates as coords
import astropy.units as u
from Survey import unweighted_survey_speed
from astropy.time import Time

SR_TO_SQ_DEG = 3282.8
SIDEREAL_DAY_SEC = 23.9344699 * 3600.

SEC2HR = 1. / 3600.
HR2SEC = 1. / SEC2HR
MIN2HR = 1. / 60.
DAY2HR = 24.
HR2DAY = 1. / DAY2HR
SEC2DAY = SEC2HR * HR2DAY
DAY2SEC = DAY2HR * HR2SEC
LUN2HR = 24 * 28.
YEAR2HR = 365.25 * 24.


class Survey():

    def __init__(self, name='',
                 lat=33.3558, fov=47., overhead=15., optimal_exposure=30.,
                 DIQ=2.0, saturation=14, limiting_mag=20.4, color='#ff7f0e',
                 dates=None):
        self.name = name
        self.lat = lat  # degrees
        self.colat = 90. - self.lat
        self.fov = fov  # square degrees
        self.overhead = overhead  # seconds
        self.optimal_exposure = optimal_exposure  # seconds
        self.DIQ = DIQ  # FWHM in arcsec
        self.saturation = saturation  # mag
        self.limiting_mag = limiting_mag
        self.color = color
        self.dates = dates  # [start year, finish year]

    def time_per_exposure(self, cadence_sec=None):
        # determine integration time:
        # max of cadence time - readout time/overhead, optimal integration time

        optimal_time_per_exposure = self.optimal_exposure + self.overhead

        if cadence_sec is not None:
            if self.overhead <= cadence_sec <= optimal_time_per_exposure:
                return cadence_sec
            elif cadence_sec > optimal_time_per_exposure:
                return optimal_time_per_exposure
            else:
                # zero integration time
                return self.overhead
        else:
            return optimal_time_per_exposure

    def areal_survey_rate(self, cadence_sec=None):
        # determine areal survey rate (square degrees/sec)
        # fov / (integration time + readout time + any overhead)

        return self.fov / self.time_per_exposure(cadence_sec=cadence_sec)

    def sky_limited_cadence_subnight(self, max_zenith_angle=66.4):
        # at what cadence (if any) do we run out of sky in a single night?
        # this is effectively for two observations per night

        omega_dot = self.areal_survey_rate()
        dov_dt = dsky_dt(self.lat, max_zenith_angle)

        if omega_dot >= dov_dt:    # camera surveys faster than sky rotates out
            snapshot_area = instantaneous_sky_area(max_zenith_angle)
            t_limit = snapshot_area / omega_dot
        else:  # Are there any possibile times with t_limit < t_night/2 but
                # omega_dot < dov_dt?  No!  so I don't need the code below
                # limit is za=90, doy=0, omega_dot=dov_dt.
                # t_limit above is always longer than hours of darkness/2.

            # TODO: remove this code, refactor into multi-night
            # have to use a subset of the sky. consider the non-circumpolar
            # area
            t_limit = average_time_above_zenith_angle(self.lat,
                                                      max_zenith_angle)
            snapshot_area = omega_dot * t_limit
            circ = circumpolar_area(self.lat, max_zenith_angle)

            # if there's more area in the circumpolar region, use that
            if circ > snapshot_area:
                snapshot_area = circ
                t_limit = snapshot_area / omega_dot

            # TODO: make sure we don't use this
            t_limit = N.inf

        return t_limit * SEC2DAY, snapshot_area

    def _total_dark_sec(self, t, doy=0):
        # how many seconds of darkness are there in a cadence period
        # t in seconds?

        dh = lambda t: N.mean(hours_of_darkness(doy +
                                                N.arange(N.ceil(t * SEC2HR / 24.)), latitude=self.lat))
        ds = lambda t: dh(t) * HR2SEC

        # total darkness: whole nights plus leftover
        return (ds(t) * N.floor(t * SEC2HR / 24.) + N.min([t % DAY2SEC, ds(t)]))

    def sky_limited_cadence_multinight(self, doy=0, max_zenith_angle=66.4):
        # at what cadence do we run out of sky over multiple nights?

        # how many hours of darkness are there in a cadence period
        # t in seconds?
        dh = lambda t: N.mean(hours_of_darkness(doy +
                                                N.arange(N.ceil(t * SEC2HR / 24.)), latitude=self.lat))
        ds = lambda t: dh(t) * HR2SEC

        # apart from changing night length, shift in nightly available sky is
        # by difference of sideral and solar days: normalize limiting time
        # by sidereal rotation rate
        fsidereal = (DAY2SEC - SIDEREAL_DAY_SEC) / DAY2SEC

        omega_dot = self.areal_survey_rate()
        dov_dt = dsky_dt(self.lat, max_zenith_angle)

        # restrict to surveys faster than the sideral rotation rate
        assert(omega_dot > dov_dt * fsidereal)

        us = lambda t: N.mean(unique_sky_per_night(self.lat,
                                                   max_zenith_angle, dh(t)))

        # initiate the solve by neglecting sidereal rotation: how many nights to
        # scan whole sky?
        guess = us(1) / (self.areal_survey_rate() * ds(1)) * DAY2SEC

        limit_cadence = lambda t: us(t) - \
            self.areal_survey_rate() * self._total_dark_sec(t)

        # optimize for solution (limit_cadence == 0)
        t_limit = brentq(limit_cadence, 0.1 * guess, 100 * guess)

        snap_area = lambda t: self.areal_survey_rate() * self._total_dark_sec(t)

        return t_limit * SEC2DAY, snap_area(t_limit)

    def sky_limited_cadence(self, doy=0, max_zenith_angle=66.4):
        # NB the sky limited cadences need bounds checking!
        # have to verify that subnight cadences is less than 2x night length;
        # multinight has to be longer than 1 night.

        t_limit, snap_area = self.sky_limited_cadence_subnight(
            max_zenith_angle=max_zenith_angle)
        dark_sec = hours_of_darkness(doy,
                                     latitude=self.lat) * HR2SEC

        # does it take more than half a night to finish?
        if t_limit > dark_sec * SEC2DAY / 2.:
            t_limit, snap_area = self.sky_limited_cadence_multinight(
                doy=doy, max_zenith_angle=max_zenith_angle)

        return t_limit, snap_area

    def snapshot_size(self, cadence_days, doy=0, max_zenith_angle=66.4,
                      absolute_mag=-19, max_mlim=None, k_corr=None):
        # determine the area and volume covered in the specified cadence period
        # if cadence area is sky-limited,
        # increase integration time (assuming limiting mag is bkg dominated)
        # returns snapshot area (square degrees) and volume (Mpc^3)

        # check for min cadence:
        min_t = (self.optimal_exposure + self.overhead) * SEC2DAY
        if cadence_days < min_t:
            return N.NaN, N.NaN, N.NaN

        # how much total wall clock time do we have to observe?
        dark_sec = self._total_dark_sec(cadence_days * DAY2SEC, doy=doy)

        # does this cadence end in daytime?
        final_doy = (doy + N.floor(cadence_days)) % 365
        hour_of_night = (cadence_days % 1.) * DAY2HR
        if hour_of_night > hours_of_darkness(final_doy, latitude=self.lat):
            return N.NaN, N.NaN, N.NaN

        # are we restricting to brighter than a certain mlim?
        base_mlim = self.limiting_mag
        if max_mlim is not None:
            if self.limiting_mag > max_mlim:
                base_mlim = max_mlim

        # check for sky-limited cadence:
        max_t, max_s = self.sky_limited_cadence(
            max_zenith_angle=max_zenith_angle)
        # if our cadence is longer than one night, check for larger max area
        if (max_t < 1) and (cadence_days > 1):
            max_t, max_s = self.sky_limited_cadence_multinight(
                max_zenith_angle=max_zenith_angle)

        if cadence_days <= max_t:  # not sky limited
            snap_area = self.areal_survey_rate() * dark_sec
            snap_vol = unweighted_survey_speed(absolute_mag,
                                               base_mlim, self.fov,
                                               (self.optimal_exposure + self.overhead), k_corr=k_corr) * \
                dark_sec
            new_limiting_mag = base_mlim
        else:
            snap_area = max_s
            # lengthen integration time proportionally
            # number of fields
            n_fields = N.round(snap_area / self.fov)
            time_per_field = dark_sec / n_fields
            t_exp = time_per_field - self.overhead  # sec
            #assert (t_exp > self.optimal_exposure)
            # now adjust the limiting mag from the optimal exposure time
            delta_mlim = -2.5 * N.log10(N.sqrt(self.optimal_exposure / t_exp))
            new_limiting_mag = self.limiting_mag + delta_mlim
            # truncate to max_mlim if needed
            if (max_mlim is not None):
                if new_limiting_mag > max_mlim:
                    new_limiting_mag = max_mlim
            snap_vol = unweighted_survey_speed(absolute_mag,
                                               new_limiting_mag, self.fov,
                                               time_per_field, k_corr=k_corr) * dark_sec

        return snap_area, snap_vol, new_limiting_mag

    def average_yearly_exposures_per_field(self, max_zenith_angle=66.4,
                                           cadence_sec=None):
        # assume images are equally divided over visible sky, no weather
        # TODO: consider making this more sophisticated with snapshot sizes,
        # circumpolar area
        doy = N.arange(365)
        yearly_dark_sec = N.sum(hours_of_darkness(doy)) * HR2SEC
        area_surveyed_per_year = yearly_dark_sec * self.areal_survey_rate(
            cadence_sec=cadence_sec)
        # divide by area available
        return N.floor(area_surveyed_per_year /
                       unique_sky_per_year(self.lat, max_zenith_angle))

    def yearly_cadence_points(self, cadence_days, twilight=18.):
        # return an array of times separating observations starting from a
        # strict grid of cadence_days observations, then filtered for daylight

        n_grid_points = N.round(365. / cadence_days)
        solstice = Time('2015-12-22 04:49:00', scale='utc')
        deltas = N.linspace(0, n_grid_points - 1,
                            n_grid_points) * cadence_days * u.day
        times = solstice + deltas

        # TODO: use real longitute and alt coordinates here
        tel_loc = coords.EarthLocation(lat=coords.Latitude(self.lat * u.deg),
                                       lon=coords.Longitude(0 * u.deg), height=1707. * u.m)

        altazframe = coords.AltAz(obstime=times, location=tel_loc, pressure=0)
        sunaltazs = coords.get_sun(times).transform_to(altazframe)
        wdark = sunaltazs.alt < twilight * u.deg
        return times[wdark].jd


class EvryscopeSurvey(Survey):

    def __init__(self, *args, **kwargs):
        Survey.__init__(self, *args, **kwargs)

    def snapshot_size(self, cadence_days, doy=0, max_zenith_angle=66.4,
                      absolute_mag=-19, max_mlim=None, k_corr=None):
        # determine the area and volume covered in the specified cadence period
        # if cadence area is sky-limited,
        # increase integration time (assuming limiting mag is bkg dominated)
        # returns snapshot area (square degrees) and volume (Mpc^3)

        if max_zenith_angle < 60:
            raise NotImplementedError(
                'Need to scale fov for small zenith angs')

        # check for min cadence:
        min_t = (self.optimal_exposure + self.overhead) * SEC2DAY
        if cadence_days < min_t:
            return N.NaN, N.NaN, N.NaN

        # how much total wall clock time do we have to observe?
        dark_sec = self._total_dark_sec(cadence_days * DAY2SEC, doy=doy)

        # does this cadence end in daytime?
        final_doy = (doy + N.floor(cadence_days)) % 365
        hour_of_night = (cadence_days % 1.) * DAY2HR
        ratchet = N.floor(hour_of_night / 2.)  # two hour ratchet
        if hour_of_night > hours_of_darkness(final_doy, latitude=self.lat):
            return N.NaN, N.NaN, N.NaN

        open_shutter_time = dark_sec * (self.optimal_exposure /
                                        (self.optimal_exposure + self.overhead))

        if cadence_days < 1:
            # 10/28/14 notes: for Northern site, 1800 deg^2 change per ratchet
            snap_area = self.fov - 1800 * ratchet
            open_shutter_per_pointing = open_shutter_time
        else:
            snap_area = self.fov + 1800 * ratchet
            # average over for different coadd depths caused by different
            # overlap regions
            open_shutter_per_pointing = open_shutter_time * self.fov / snap_area

        # not being too careful about readnoise in these coadds
        delta_mlim = -2.5 * N.log10(N.sqrt(self.optimal_exposure /
                                           open_shutter_per_pointing))
        new_limiting_mag = self.limiting_mag + delta_mlim
        # truncate to max_mlim if needed
        if (max_mlim is not None):
            if new_limiting_mag > max_mlim:
                new_limiting_mag = max_mlim
        snap_vol = unweighted_survey_speed(absolute_mag,
                                           new_limiting_mag, snap_area,
                                           dark_sec, k_corr=k_corr) * dark_sec

        return snap_area, snap_vol, new_limiting_mag


# colors: https://github.com/mbostock/d3/wiki/Ordinal-Scales#categorical-colors

ZTF = Survey(name='ZTF', dates=[2017, 2020])
PTF = Survey(name='PTF', fov=7.26, overhead=46., optimal_exposure=60.,
             color='#ffbb78', dates=[2009, 2016])

# fiducial rather than optimal exposures below
PS1 = Survey(name='PS1', lat=20.71, fov=7., overhead=10., optimal_exposure=30.,
             saturation=13.5, limiting_mag=21.8, color='#c5b0d5', dates=[2010, 2015])
PS12 = Survey(name='PS1 & 2', lat=20.71, fov=7. * 2., overhead=10.,
              optimal_exposure=30.,
              saturation=13.5, limiting_mag=21.8, color='#c5b0d5', dates=[2015, 2018])
CRTS = Survey(name='CRTS', lat=32.42, fov=8., overhead=18., optimal_exposure=30.,
              limiting_mag=19.5, color='#98df8a', dates=[2003, 2015])
CRTS2 = Survey(name='CRTS-2', lat=32.42, fov=19., overhead=12., optimal_exposure=30.,
               limiting_mag=19.5, color='#2ca02c', dates=[2015, 2018])
DECam = Survey(name='DECam', lat=N.abs(-30.17), fov=3., overhead=20., optimal_exposure=50.,
               saturation=16, limiting_mag=23.7, color='#d62728', dates=[2013, 2018])
# assume Haleakala
ATLAS = Survey(name='ATLAS', lat=20.71, fov=60., overhead=8., optimal_exposure=30.,
               saturation=12.5, limiting_mag=20.0, color='#9467bd', dates=[2016, 2019])
BlackGEM4 = Survey(name='BlackGEM-4', lat=N.abs(-29.26), fov=11., overhead=5., optimal_exposure=30.,
                   limiting_mag=20.7, color='#c7c7c7', dates=[2017, 2021])
LSST = Survey(name='LSST', lat=N.abs(-30.24), fov=9.6, overhead=11., optimal_exposure=30.,
              saturation=16, limiting_mag=24.7, color='#1f77b4', dates=[2021, 2031])
# at CTIO.  4 sec full frame readout, no slews.
# anti-blooming, so don't really saturate (per NL email, 10/8/14)
EvryScope = EvryscopeSurvey(name='Evryscope', lat=N.abs(-30.17), fov=8660, overhead=4., optimal_exposure=120.,
                            saturation=1, limiting_mag=16.4, color='#17becf', dates=[2015, 2018])
# HSC ETC: https://hscq.naoj.hawaii.edu/cgi-bin/HSC_ETC/hsc_etc.cgi
# I chose 25 mag as a target, got
HSC = Survey(name='HSC', lat=19.83, fov=N.pi * (45. / 60.)**2., overhead=40.,
             optimal_exposure=100., saturation=18.6, limiting_mag=25, color='#660066', dates=[2014, 2020])
# lots of others I could add: MASTER, ROTSE, Skymapper, etc.

# fov for single four-tel site; lat is Haleakala
# fov, overhead, etc. from B. Shappee email 8/26/15
ASASSN = Survey(name='ASAS-SN 1', lat=20.71, fov=73., overhead=23., optimal_exposure=180.,
                saturation=8, limiting_mag=17, color='#990000', dates=[2013, 2017])

surveys = [PTF, ZTF, PS1, PS12, BlackGEM4, CRTS,
           CRTS2, DECam, ATLAS, LSST, EvryScope, ASASSN, HSC]


def airmass_to_zenith_angle(airmass):
    return N.degrees(N.arccos(1. / airmass))


def zenith_angle_to_airmass(zenith_angle):
    return 1. / N.cos(N.radians(zenith_angle))


def hours_above_zenith_angle(latitude, declination, zenith_angle):
    # inputs in degrees
    # gives nans for values that are circumpolar and/or never rise above
    # the zenith angle
    # Smart II.3
    l = N.radians(latitude)
    d = N.radians(declination)
    z = N.radians(zenith_angle)
    H = N.arccos((N.cos(z) - N.sin(l) * N.sin(d)) /
                 (N.cos(l) * N.cos(d)))  # hour angle in radians

    return 2. * H / (2. * N.pi) * SIDEREAL_DAY_SEC * SEC2HR


# Define a function which returns the hours of daylight
# given the day of the year, from 0 to 365
# https://jakevdp.github.io/blog/2014/06/10/is-seattle-really-seeing-an-uptick-in-cycling/
def hours_of_daylight(date, axis=23.44, latitude=33.356):
    """Compute the hours of daylight for the given date"""
    diff = date - pd.datetime(2000, 12, 21)
    day = diff.total_seconds() / 24. / 3600
    doy %= 365.25
    m = 1. - N.tan(N.radians(latitude)) * N.tan(N.radians(axis) *
                                                N.cos(doy * N.pi / 182.625))
    m = max(0, min(m, 2))
    return 24. * N.degrees(N.arccos(1 - m)) / 180.

# http://www.gandraxa.com/length_of_day.xml


def hours_of_darkness(doy, axis=23.44, latitude=33.356, twilight=18.):
    """Compute the hours of darkness (greater than t degree twilight) 
        for the given day of year relative to winter solstice"""
#    diff = date - pd.datetime(2000, 12, 21)
#    day = diff.total_seconds() / 24. / 3600
#    doy %= 365.25

    if type(doy) in [int, float, N.int, N.float, N.float64]:
        doy = N.array([doy])

    m = 1. - N.tan(N.radians(latitude)) * N.tan(N.radians(axis) *
                                                N.cos(doy * N.pi / 182.625))
    i = N.tan(N.radians(twilight)) / N.cos(N.radians(latitude))
    n = m + i
    #n = N.max(0, N.min(n, 2))
    # vectorize
    n[n > 2] = 2
    n[n < 0] = 0
    return 24. * (1. - N.degrees(N.arccos(1 - n)) / 180.)


def dsky_dt(latitude, max_zenith_angle):
    # change in unique sky (in square degrees) above max_zenith_angle per second
    # calculated by determining steradians/sec transiting the meridian above
    # max_zenith_angle, excluding the circumpolar region

    SIDEREAL_ROTATION_RATE = 2 * N.pi / SIDEREAL_DAY_SEC  # rad/sec

    theta1, theta2 = non_circumpolar_limits(latitude, max_zenith_angle)
    # integrate in dec along the meridan
    return (N.cos(N.radians(theta1)) - N.cos(N.radians(theta2))) * \
        SIDEREAL_ROTATION_RATE * SR_TO_SQ_DEG


def non_circumpolar_limits(latitude, max_zenith_angle):
    # return angles theta1, theta2 from north pole: theta1 is at
    # airmass cut or edge of circumpolar circle, theta2 is at airmass cut
    assert(max_zenith_angle <= 90.)
    assert(90. >= latitude >= -90.)

    # make latitude positive.  Doesn't change the calculation, but simplifies
    # code logic
    if latitude < 0:
        latitude *= -1.

    colatitude = 90. - latitude

    # dec of zenith is latitude; find integration limits
    if (colatitude - max_zenith_angle) >= 0:  # no circumpolar area
        theta1 = colatitude - max_zenith_angle
        theta2 = colatitude + max_zenith_angle
    else:
        # only include area outside of the circumpolar area
        theta1 = max_zenith_angle - colatitude
        theta2 = colatitude + max_zenith_angle

    assert (theta2 >= theta1)

    return theta1, theta2


def average_time_above_zenith_angle(latitude, max_zenith_angle):
    # area weighted time above zenith angle for non-circumpolar sky
    # S theta sin[theta] dtheta / S sin[theta] dtheta

    theta1, theta2 = non_circumpolar_limits(latitude, max_zenith_angle)

    rt1 = N.radians(theta1)
    rt2 = N.radians(theta2)

    rlat = N.radians(N.abs(latitude))

    rtavg = (rt1 * N.cos(rt1) - rt2 * N.cos(rt2) - N.sin(rt1) +
             N.sin(rt2)) / (N.cos(rt1) - N.cos(rt2))

    # Smartt II.3: hour angle of a zenith angle
    HA = N.arccos((N.cos(N.radians(max_zenith_angle)) -
                   N.sin(rlat) * N.sin(rtavg)) / (N.cos(rlat) * N.cos(rtavg)))

    return 2 * HA / (2. * N.pi) * SIDEREAL_DAY_SEC


def circumpolar_area(latitude, max_zenith_angle):
    colatitude = 90. - latitude
    # TODO: check for Southern colats
    if (colatitude - max_zenith_angle) >= 0:  # no circumpolar area
        return 0.
    else:
        return 2 * N.pi * (1 - N.cos(N.radians(max_zenith_angle - colatitude))) * \
            SR_TO_SQ_DEG


def unique_sky_per_year(latitude, max_zenith_angle):
    # notes, 1/14/13
    colatitude = 90. - latitude
    # TODO: check for Southern colats
    if (colatitude - max_zenith_angle) >= 0:  # no circumpolar area
        return 4 * N.pi * N.sin(N.radians(colatitude)) * \
            N.sin(N.radians(max_zenith_angle)) * SR_TO_SQ_DEG
    else:
        return 2 * N.pi * (1 - N.cos(N.radians(colatitude + max_zenith_angle))) * \
            SR_TO_SQ_DEG


# test:
# dsky_dt(lat, zmax)*SIDEREAL_DAY_SEC + circumpolar_area(lat, zmax) ==
# unique_sky_per_year(lat, zmax)
# for all lats, zmaxes


# max_dark_hours = hours_of_darkness(0, latitude=latitude,
#        twilight=twilight)
# min_dark_hours = hours_of_darkness(183, latitude=latitude,
#        twilight=twilight)
def unique_sky_per_night(latitude, max_zenith_angle, night_length_hours,
                         twilight=18.):

    if night_length_hours > 0:
        return instantaneous_sky_area(max_zenith_angle) + \
            dsky_dt(latitude, max_zenith_angle) * night_length_hours * 3600.
    else:
        return 0.


def instantaneous_sky_area(zenith_angle, degrees=True):
    """Solid angle of sky above a minimum zenith angle (in degrees)"""
    # calculations in May 16th notebook
    if degrees:
        units = SR_TO_SQ_DEG
    else:
        units = 1.
    return 2. * N.pi * (1. - N.cos(N.radians(zenith_angle))) * units


def linear_control_time(absolute_mag, limiting_mag, tau_eff, z, k_corr=None):
    # estimate of control time from simple light curve model: see 8/18/15 notes
    # absolute_mag is peak absolute mag
    # tau_eff is days to decline 1 mag
    # returns control time in days

    DM = cp.magnitudes.distance_modulus(z, **cp.fidcosmo)

    # use k-correction for an f_lambda standard: k = -2.5 log10(1./(1+z))
    # see Hogg 99 eqn. 27

    if k_corr is None:
        k_corr = lambda z: -2.5 * N.log10(1. / (1. + z))

    apparent_mag_peak = absolute_mag + DM + k_corr(z)

    ct = -1. * (apparent_mag_peak - limiting_mag) * tau_eff * (1. + z)

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

    return (control_time_days + N.sum(dt_obs_days[w_dt_lt]) +
            N.sum(~w_dt_lt) * control_time_days) / 365.25


def sum_control_time_k_consecutive(obs_points_jd, control_time_days,
                                   k_consecutive):
    # control time is in observer frame
    # returns years

    dt_obs_days = obs_points_jd[(k_consecutive - 1):] - \
        obs_points_jd[:-(k_consecutive - 1)]

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

    return ct_sum / 365.25


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
                          rate_z=None, k_corr=None, max_mlim=None,
                          max_zenith_angle=66.4, k_consecutive=1):
    # rate a function of z in events Mpc^-3 yr^-1
    # for efficiency, require scalar cadences

    obs_points_jd = survey.yearly_cadence_points(cadence_days)

    if rate_z is None:
        # use the Ia rate by default (see LSST SB)
        rate_z = lambda z: 3.E-5

    doy = N.arange(365)

    n_events = N.zeros([absolute_mags.size, tau_effs.size])

    for i, absolute_mag in enumerate(absolute_mags):
        # we're being a little fast and loose here with the snapshot/all-sky
        # distinction
        area, vol, mlim = zip(*[survey.snapshot_size(cadence_days,
                         doy=d, max_zenith_angle=max_zenith_angle,
                         absolute_mag=absolute_mag, max_mlim=max_mlim,
                         k_corr=k_corr) for d in doy])

        snap_area = N.mean(area)
        limiting_mag = N.mean(mlim)

        z_limit = limiting_z(limiting_mag, absolute_mag, k_corr=k_corr)

        for j, tau_eff in enumerate(tau_effs):
            def integrand(z):
                ctz = linear_control_time(absolute_mag, limiting_mag,
                                          tau_eff, z, k_corr=k_corr)
                return rate_z(z) / (1. + z) * \
                    sum_control_time(obs_points_jd, ctz,
                                     k_consecutive=k_consecutive) * \
                    snap_area / SR_TO_SQ_DEG * \
                    cp.distance.diff_comoving_volume(z, **cp.fidcosmo)

            n_events[i, j] = quad(integrand, 0, z_limit)[0]
            print absolute_mag, tau_eff, n_events[i, j]

    return n_events
