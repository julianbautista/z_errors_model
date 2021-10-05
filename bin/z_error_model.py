import argparse, time, pickle 
import numpy as np
import matplotlib.pyplot as plt
from numba import jit, prange
from astropy.cosmology import Planck18 as cosmo
from astropy.io import fits

def read_gamma_data(file_wo_errors='delta_attributes_boring.fits.gz',
                    file_with_errors='delta_attributes_boring_500.fits.gz'):
    ''' Create gamma = C1/C0 - 1 from mean continua from mocks 
    '''

    d0 = fits.open(file_wo_errors)[3].data
    d1 = fits.open(file_with_errors)[3].data

    lambda_rest_tab = 10**d0.loglam_rest
    gamma_tab = d0.mean_cont/d1.mean_cont - 1
    return lambda_rest_tab.astype(float), gamma_tab.astype(float)

def read_z_dist_pixels(input_file='delta_attributes_boring.fits.gz', 
    get_xi_weight=False):
    ''' Get weighted pixel distribution 
    '''

    d0 = fits.open(input_file)[1].data
    
    pix_lambda = 10**d0.loglam
    pix_z = pix_lambda/1216. - 1
    pix_dist = d0.weight
    
    if get_xi_weight:
        pix_dist *= ((1+pix_z)/(1+2.5))**2.9

    pix_dist /= np.trapz(pix_dist, x=pix_z)
    return pix_z.astype(float), pix_dist.astype(float) 


def read_z_dist_quasars(input_file='zcat_drq.fits',
    get_xi_weight=False):
    ''' Get normalised redshift distribution of quasars 
    '''
    q = fits.open(input_file)[1].data
    qso_dist, z_edges = np.histogram(q.Z, bins=100, density=True)
    qso_z = 0.5*(z_edges[:-1]+z_edges[1:])

    if get_xi_weight:
        qso_dist *= ((1+qso_z)/(1+2.5))**1.44

    return qso_z.astype(float), qso_dist.astype(float)

def read_xi_qq(input_file='xi_qq_model_lyacolore.fits'):
    a = fits.open(input_file)[1].data 
    xiqq_r = a.r.astype(float)
    xiqq_xi = a.xi.astype(float)
    return xiqq_r, xiqq_xi

@jit(nopython=True)
def get_cross_fake(zq1, zq2, r_trans, r_parallel,
    z_cosmo, r_cosmo,
    lambda_rest_tab, gamma_tab,
    pix_z, pix_dist,
    xiqq_r, xiqq_xi,
    lambda_obs_min, lambda_obs_max, 
    lambda_rest_min, lambda_rest_max):
    ''' Computes the product of xi_qso x gamma for a given configuration
        zq1 : redshift of quasar 1
        zq2 : redshift of quasar 2 
        r_trans : transverse separation between pixel in forest 2 and quasar 1
        r_parallel : parallel separation between pixel in forest 2 and quasar 1
    '''
    #-- Comoving distance of quasars
    rq1 = np.interp(zq1,  z_cosmo,  r_cosmo)
    rq2 = np.interp(zq2,  z_cosmo,  r_cosmo)

    #-- Get value of QSO x QSO correlation function
    r_qso_parallel = rq2-rq1 
    r = np.sqrt(r_trans**2+r_qso_parallel**2)
    xi = np.interp(r, xiqq_r, xiqq_xi)
    
    #-- Convert r_parallel to rest-frame wavelength
    r_pixel = rq1 + r_parallel 
    z_pixel = np.interp(r_pixel,  r_cosmo,  z_cosmo)
    lambda_obs = 1216.*(1+z_pixel)
    lambda_rest = lambda_obs/(1+zq2)
    
    if (lambda_obs <  lambda_obs_min or 
        lambda_obs >  lambda_obs_max or
        lambda_rest <  lambda_rest_min or
        lambda_rest >  lambda_rest_max): 
        weight = 0
        return 0, 0
    else:
        weight =  np.interp(z_pixel, pix_z, pix_dist)

    #-- Get value of contamination at this rest-frame wavelength
    gam =  np.interp(lambda_rest, lambda_rest_tab, gamma_tab)

    return gam * xi, weight

@jit(nopython=True)
def create_cross(rt_nbins, rp_nbins, rt_max, rp_max,
    z_qso_min, z_qso_max, z_nbins,
    lambda_obs_min, lambda_obs_max, 
    lambda_rest_min, lambda_rest_max,
    z_cosmo, r_cosmo,
    lambda_rest_tab, gamma_tab,
    qso_z, qso_dist,
    pix_z, pix_dist,
    xiqq_r, xiqq_xi):
    ''' Computes the contamination for several values of (rt, rp)
    '''

    rt_edges = np.linspace(0, rt_max, rt_nbins+1)
    rp_edges = np.linspace(-rp_max, 0, rp_nbins+1)
    rt_grid = (rt_edges[1:]+rt_edges[:-1])*0.5
    rp_grid = (rp_edges[1:]+rp_edges[:-1])*0.5

    z_values = np.linspace(z_qso_min, z_qso_max, z_nbins)

    xi_grid = np.zeros((rt_nbins, rp_nbins))
    we_grid = np.zeros((rt_nbins, rp_nbins))

    #-- Loop over separation bins
    for i in range(rt_nbins):
        rt = rt_grid[i]
        for j in range(rp_nbins):
            rp = rp_grid[j]

            #-- Loop over the first quasar
            for zq1 in z_values:
                if zq1 < z_qso_min or zq1 > z_qso_max: 
                    continue 
                we_qso_1 =  np.interp(zq1, qso_z, qso_dist)

                #-- Loop over the second quasar
                for zq2 in np.linspace(zq1-0.2, zq1+0.2, z_nbins):
                    if zq2 < z_qso_min or zq2 > z_qso_max: 
                        continue
                    we_qso_2 =  1 # np.interp(zq2, qso_z, qso_dist)
                    xi, we =  get_cross_fake(zq1, zq2, rt, rp, 
                                z_cosmo, r_cosmo,
                                lambda_rest_tab, gamma_tab,
                                pix_z, pix_dist,
                                xiqq_r, xiqq_xi,
                                lambda_obs_min, lambda_obs_max, 
                                lambda_rest_min, lambda_rest_max)
                    xi_grid[i, j] += xi * we * we_qso_1 * we_qso_2
                    we_grid[i, j] += we * we_qso_1 * we_qso_2

    for i in range(rt_nbins):
        for j in range(rp_nbins):
            if we_grid[i, j] > 0:
                xi_grid[i, j] /= we_grid[i, j]

    return rt_edges, rp_edges, xi_grid, we_grid

@jit(nopython=True)
def get_auto_fake(zq1, zq2, r_trans, r_parallel, lambda_rest_1,
    z_cosmo, r_cosmo,
    lambda_rest_tab, gamma_tab,
    pix_z, pix_dist,
    xiqq_r, xiqq_xi,
    lambda_obs_min, lambda_obs_max, 
    lambda_rest_min, lambda_rest_max):
    ''' Computes the product of xi_qso x gamma for a given configuration
        zq1 : redshift of quasar 1
        zq2 : redshift of quasar 2 
        r_trans : transverse separation between pixel in forest 2 and quasar 1
        r_parallel : parallel separation between pixel in forest 2 and quasar 1
        lambda_rest_1 : rest-frame wavelength of pixel in forest 1
    '''
    rq1 =  np.interp(zq1,  z_cosmo,  r_cosmo)
    rq2 =  np.interp(zq2,  z_cosmo,  r_cosmo)

    #-- Get value of QSO x QSO correlation function
    r_qso_parallel = rq2-rq1 
    r = np.sqrt(r_trans**2+r_qso_parallel**2)
    xi =  np.interp(r, xiqq_r, xiqq_xi)

    #-- Convert r_parallel to rest-frame wavelength
    lambda_obs_1 = lambda_rest_1*(1+zq1)
    z_pixel_1 = lambda_obs_1/1216. - 1

    if (lambda_obs_1 <  lambda_obs_min or 
        lambda_obs_1 >  lambda_obs_max or
        lambda_rest_1 <  lambda_rest_min or
        lambda_rest_1 >  lambda_rest_max): 
        weight = 0
    else:
        weight = np.interp(z_pixel_1, pix_z, pix_dist)

    gamma_1 = np.interp(lambda_rest_1, lambda_rest_tab, gamma_tab)
    r_pixel_1 = np.interp(z_pixel_1,  z_cosmo,  r_cosmo)

    #-- Compute lambda_rest_2 from r_pixel_1, r_paralle, z_pixel_2
    r_pixel_2 = r_pixel_1 + r_parallel 
    z_pixel_2 = np.interp(r_pixel_2,  r_cosmo,  z_cosmo)
    lambda_obs_2 = 1216.*(1+z_pixel_2)
    lambda_rest_2 = lambda_obs_2/(1+zq2)
    

    if (lambda_obs_2 <  lambda_obs_min or 
        lambda_obs_2 >  lambda_obs_max or
        lambda_rest_2 <  lambda_rest_min or
        lambda_rest_2 >  lambda_rest_max): 
        weight *= 0
    else: 
        weight *=  np.interp(z_pixel_2, pix_z, pix_dist)

    gamma_2 =  np.interp(lambda_rest_2, lambda_rest_tab, gamma_tab)
    
    return gamma_1 * gamma_2 * xi, weight

@jit(nopython=True, parallel=True)
def create_auto(rt_nbins, rp_nbins, rt_max, rp_max,
    z_qso_min, z_qso_max, z_nbins,
    lambda_obs_min, lambda_obs_max, 
    lambda_rest_min, lambda_rest_max,
    lambda_rest_nbins,
    z_cosmo, r_cosmo,
    lambda_rest_tab, gamma_tab,
    qso_z, qso_dist,
    pix_z, pix_dist,
    xiqq_r, xiqq_xi
    ):

    rt_edges = np.linspace(0, rt_max, rt_nbins+1)
    rp_edges = np.linspace(0, rp_max, rp_nbins+1)
    rt_grid = (rt_edges[1:]+rt_edges[:-1])*0.5
    rp_grid = (rp_edges[1:]+rp_edges[:-1])*0.5

    z_values = np.linspace(z_qso_min, z_qso_max, z_nbins)
    lambda_rest_values = 10**np.linspace(np.log10(lambda_rest_min),  
                                         np.log10(lambda_rest_max),
                                         lambda_rest_nbins)

    xi_grid = np.zeros((rt_nbins, rp_nbins))
    we_grid = np.zeros((rt_nbins, rp_nbins))

    #-- Loop over separation bins
    for i in prange(rt_nbins):
        rt = rt_grid[i]
        for j in range(rp_nbins):
            rp = rp_grid[j]

            #-- Loop over the first quasar
            for zq1 in z_values:
                if zq1 < z_qso_min or zq1 > z_qso_max: 
                    continue 
                we_qso_1 = np.interp(zq1, qso_z, qso_dist)

                #-- Loop over the second quasar
                for zq2 in np.linspace(zq1-0.2, zq1+0.2, z_nbins):
                    if zq2 < z_qso_min or zq2 > z_qso_max: 
                        continue
                    we_qso_2 = 1 # np.interp(zq2, qso_z, qso_dist)

                    #-- Loop over lambda_rest_1
                    #-- The lambda_rest_2 is
                    #-- computed from (zq1, zq2, rt, rp, lambda_rest_1)
                    for lambda_rest_1 in lambda_rest_values:
                        xi, we = get_auto_fake(
                                    zq1, zq2, rt, rp, lambda_rest_1,
                                    z_cosmo, r_cosmo,
                                    lambda_rest_tab, gamma_tab,
                                    pix_z, pix_dist,
                                    xiqq_r, xiqq_xi,
                                    lambda_obs_min, lambda_obs_max, 
                                    lambda_rest_min, lambda_rest_max
                                    )
                        xi_grid[i, j] += xi * we * we_qso_1 * we_qso_2 
                        we_grid[i, j] += we * we_qso_1 * we_qso_2

    for i in range(rt_nbins):
        for j in range(rp_nbins):
            if we_grid[i, j] > 0:
                xi_grid[i, j] /= we_grid[i, j]

    return rt_edges, rp_edges, xi_grid, we_grid

def plot_xi(rt=None, rp=None, xi=None, we=None):

    plt.figure()
    vmax = np.max(np.abs(xi))
    plt.pcolormesh(rt, rp, xi.T, cmap='seismic', vmin=-vmax, vmax=vmax)
    plt.colorbar()
    plt.xlabel(r'$r_\perp$ [$h^{-1}$ Mpc]')
    plt.ylabel(r'$r_\parallel$ [$h^{-1}$ Mpc]')

def plot_data(data):

    for key in ['cross', 'auto']:
        if key in data:
            d = data[key]
            plot_xi(rt=d['rt'], rp=d['rp'], xi=d['xi'])
            plt.title(key)


def save_to_pickle(data, filename):
    pickle.dump(data, open(filename, 'wb'))

def load_from_pickle(filename):
    data = pickle.load(open(filename, 'rb'))
    return data

def compute_model(args):
    
    #-- Create interpolator for redshift - distance relation
    #-- Create fiducial cosmology
    h = cosmo.H0.value/100.
    z_cosmo = np.linspace(2., 4., 100)
    r_cosmo = cosmo.comoving_distance(z_cosmo)*h 

    #-- Read gamma function
    lambda_rest_tab, gamma_tab = read_gamma_data(
        file_wo_errors   = args.meancont_no_errors,
        file_with_errors = args.meancont_with_errors)

    #-- Read pixel weight distribution
    pix_z, pix_dist = read_z_dist_pixels(
        input_file = args.pixel_dist, get_xi_weight=True)

    #-- Read quasar distribution
    qso_z, qso_dist = read_z_dist_quasars(
        input_file = args.qso_catalog, get_xi_weight=True)

    #-- Read qso auto-correlation 
    xiqq_r, xiqq_xi = read_xi_qq(input_file=args.xiqq)

    data = {}

    #-- Compute cross-correlation
    t0 = time.time()
    rtc, rpc, xic, wec = create_cross(
        args.rt_nbins, args.rp_nbins, args.rt_max, args.rp_max,
        args.z_qso_min, args.z_qso_max, args.z_nbins,
        args.lambda_obs_min, args.lambda_obs_max, 
        args.lambda_rest_min, args.lambda_rest_max,
        z_cosmo, r_cosmo,
        lambda_rest_tab, gamma_tab,
        qso_z, qso_dist,
        pix_z, pix_dist,
        xiqq_r, xiqq_xi
    )
    t1 = time.time() 
    dt = (t1-t0)/60.
    print(f'Time elapsed with cross: {dt:.1f} minutes')
    data['cross'] = {'rt': rtc, 'rp': rpc, 'xi': xic, 'we': wec}

    #-- Compute auto-correlation
    if args.skip_auto == False:
        t1 = time.time()
        rta, rpa, xia, wea = create_auto(
            args.rt_nbins, args.rp_nbins, args.rt_max, args.rp_max,
            args.z_qso_min, args.z_qso_max, args.z_nbins,
            args.lambda_obs_min, args.lambda_obs_max, 
            args.lambda_rest_min, args.lambda_rest_max,
            args.lambda_rest_nbins,
            z_cosmo, r_cosmo,
            lambda_rest_tab, gamma_tab,
            qso_z, qso_dist,
            pix_z, pix_dist,
            xiqq_r, xiqq_xi
            )
        t2 = time.time()
        dt = (t2-t1)/60. 
        print(f'Time elapsed with auto: {dt:.1f} minutes')
        data['auto'] = {'rt': rta, 'rp': rpa, 'xi': xia, 'we': wea}

    data['parameters'] = vars(args)
    save_to_pickle(data, args.output)

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--z-qso-min', type=float, default=2.1)
    parser.add_argument('--z-qso-max', type=float, default=3.5)
    parser.add_argument('--z-nbins', type=int, default=50)
    parser.add_argument('--lambda-rest-min', type=float, default=1040.)
    parser.add_argument('--lambda-rest-max', type=float, default=1200.)
    parser.add_argument('--lambda-obs-min', type=float, default=3600.)
    parser.add_argument('--lambda-obs-max', type=float, default=5400.)
    parser.add_argument('--lambda-rest-nbins', type=int, default=50)
    parser.add_argument('--rt-nbins', type=int, default=10)
    parser.add_argument('--rp-nbins', type=int, default=40)
    parser.add_argument('--rt-max', type=float, default=40.)
    parser.add_argument('--rp-max', type=float, default=200.)
    parser.add_argument('--skip-auto', action='store_true', default=False)
    parser.add_argument('--meancont-no-errors', type=str, default='etc/delta_attributes_boring.fits.gz')
    parser.add_argument('--meancont-with-errors', type=str, default='etc/delta_attributes_boring_500.fits.gz')
    parser.add_argument('--pixel-dist', type=str, default='etc/delta_attributes_boring.fits.gz')
    parser.add_argument('--qso-catalog', type=str, default='etc/zcat_drq.fits')
    parser.add_argument('--xiqq', type=str, default='etc/xi_qq_model_lyacolore.fits')
    parser.add_argument('--output', type=str, default='z_error_model.pkl')
    parser.add_argument('--plot', type=str)
    #args = vars(parser.parse_args())
    args = parser.parse_args()

    if not args.plot is None:
        data = load_from_pickle(args.plot)
        plot_data(data)
        plt.show()
    else:
        compute_model(args)
