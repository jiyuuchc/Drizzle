import numpy as np
import math
from scipy.special import erf
from scipy.ndimage import zoom
from scipy.ndimage import uniform_filter
from scipy.optimize import curve_fit

def gauss_psf(sigma, psf_size = 0):
    if psf_size == 0:
        psf_size = math.ceil(sigma * 3)

    psf = np.exp(-np.arange(-psf_size,psf_size+1, dtype=np.float32)**2 / 2 / sigma / sigma)
    psf /= psf.sum()

    psf  = psf[..., np.newaxis] * psf[np.newaxis, ...]
    return psf

def erf_psf(sigma, psf_size = 0):
    if psf_size == 0:
        psf_size = math.ceil(sigma * 3)

    x = (np.arange(-psf_size, psf_size + 2, dtype=np.float32) - 0.5)
    y = np.diff(erf(x / math.sqrt(2) / sigma))
    y /= y.sum()
    psf = y[..., np.newaxis] * y[np.newaxis,...]
    return psf

def adjusted_psf_subpixel(psf_in, zoom, scale_psf = False):
    ''' Adjust psf to accormodate subpixel computations.
    Input
        psf_in: orginal 2D psf
        zoom: integer representing size ratio of original pixel to subpixel
        scale_psf: boolean. If true, the psf_in was in original pixel size scale; otherwise, it is in subpixel size.

    Output:
        psf_out: the adjusted psf at subpixel size
    '''
    if scale_psf:
        psf_out = zoom(psf_in, zoom)
    else:
        fs = psf_in.shape[0]
        hfs = math.ceil(fs/2/zoom - 0.5)
        dfs = (2 * hfs + 1) * zoom - fs
        dfs1 = dfs // 2
        dfs2 = dfs - dfs1
        psf_out = np.pad(psf_in, ((dfs1, dfs2),(dfs1, dfs2)))

    psf_out = uniform_filter(psf_out, size=zoom, mode='constant')
    psf_out /= psf_out.sum()

    return psf_out

def analyze_flux(cnts):
    #cnts = np.sum(imgs, axis = (1,2))
    x = np.arange(cnts.size)

    def _exp_func(x,a,b,k):
        return a * np.exp(-k*x) + b

    pars, _ = curve_fit(_exp_func, x, cnts, [cnts[0],0.0,1.0])
    vals = _exp_func(x, *pars)
    dvals = -np.diff(vals)

    x_eq = pars[1]
    x_0 = pars[0] - x_eq
    kn = pars[2] * x_eq / x_0
    kp = pars[2] * (x_0 - x_eq) / x_0
    jn = (x_0 - x_eq) / x_0 * dvals + kp * x_eq
    jp = kn * (x_0 - x_eq) - x_eq / x_0 * dvals
    flux = (jn+jp) / vals[:-1]

    return flux
