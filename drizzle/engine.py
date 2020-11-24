import cupy as cp
from cupy.lib import stride_tricks
import numpy as np

_default_config = {
    "n_samples": 1000,
    "n_skip": 10,
    "n_burn_in": 200,
    "map_iterations": 500,
    "uninformative_a": 1.0,
    "brightness": 1000,
}

def update_config(config):
    _default_config.update(config)

def sample_one_image(theta, filter2, y, alpha = 1.0, beta = 0.0):
    '''
    theta: proposal emitting rate. cupy array (h x w)
    filter2: PSF. cupy array (fs x fs)
    y: photon counts. cupy array (h x w)
    alpha, beta: prior
    '''

    h, w = theta.shape
    fs = filter2.shape[0]
    hfs = fs // 2

    assert(hfs * 2 + 1 == fs)

    theta_new = cp.zeros_like(theta)
    theta = theta[cp.newaxis, cp.newaxis, ...] * filter2[..., cp.newaxis, cp.newaxis] ## array of [fs, fs, h, w]

    orig_strides = theta.strides
    new_strides = orig_strides[0] + orig_strides[2], orig_strides[1] + orig_strides[3], orig_strides[2], orig_strides[3]
    theta_rolled = stride_tricks.as_strided(theta, shape = (fs, fs, h - hfs * 2, w - hfs * 2), strides = new_strides)

    theta_sum = cp.sum(theta_rolled, axis = (0,1))
    y_remain = y[hfs:h-hfs, hfs:w-hfs].copy()

    for idx in np.arange(fs * fs - 1):
        i,j = np.unravel_index(idx, (fs,fs))
        tmp = cp.random.binomial(y_remain, theta_rolled[i,j,...] / theta_sum)
        y_remain -= tmp
        theta_new[i : i + h - hfs * 2, j : j + w - hfs * 2] += tmp
        theta_sum -= theta_rolled[i, j, ...]
    theta_new[hfs*2:h, hfs*2:w] += y_remain

    return cp.random.gamma(theta_new + alpha, 1.0 / (beta + 1))

def map_iteration(theta, filter2, y, alpha = 1.0, beta = 0.0):
    '''
    theta: proposal emitting rate. cupy array (h x w)
    filter2: PSF. cupy array (fs x fs)
    y: photon counts. cupy array (h x w)
    alpha, beta: prior
    '''

    h, w = theta.shape
    hs, ws = y.shape
    zoom = h // hs
    fs = filter2.shape[0]
    hfs = fs // zoom // 2

    assert(fs == filter2.shape[1])
    assert(hfs * 2 * zoom + zoom == fs)
    assert(zoom * hs == h and zoom * ws == w)

    ## array of [fs//zoom, hs, zoom, fs//zoom, ws, zoom]
    theta_p = theta.reshape(1, hs, zoom, 1, ws, zoom) * filter2.reshape(fs//zoom, 1, zoom, fs//zoom, 1, zoom)

    orig_strides = theta_p.strides
    new_strides = (
        orig_strides[0]+orig_strides[1],
        orig_strides[1],
        orig_strides[2],
        orig_strides[3]+orig_strides[4],
        orig_strides[4],
        orig_strides[5],
        )
    theta_rolled = stride_tricks.as_strided(
        theta_p,
        shape=(fs//zoom, hs - hfs * 2, zoom, fs//zoom, ws - hfs * 2, zoom),
        strides = new_strides
        )

    # Whelp, cupy does not support 'where' in ufunc
    # cp.true_divide(theta_rolled, theta_sum, out = theta_rolled, where = theta_sum != 0)
    theta_sum = theta_rolled.sum(axis=(0,2,3,5), keepdims=True)
    theta_rolled /= theta_sum
    theta_rolled[~cp.isfinite(theta_rolled)] = 0.0
    theta_rolled *= y[cp.newaxis, hfs:hs-hfs, cp.newaxis, cp.newaxis, hfs:ws-hfs, cp.newaxis]

    theta[...] = 0.0
    theta = theta.reshape(hs, zoom, ws, zoom)
    for i,j in np.ndindex(fs//zoom, fs//zoom):
        theta[i : i + hs - hfs * 2, :, j : j + ws - hfs * 2, :] += theta_rolled[i, :, :, j, :, :]
    theta = theta.reshape(h,w)

    theta += alpha - 1.0
    theta.clip(0, out = theta)

    if not cp.isscalar(beta):
        theta /= beta + 1.0

    return theta

def sparse_deconv(img, psf, zoom = 1.0, alpha = 0.0):
    img = cp.array(img)
    psf = cp.array(psf)

    hs, ws = img.shape
    theta = cp.ones((hs * zoom, ws * zoom), dtype=cp.float32)
    for i in range(_default_config["map_iterations"]):
        theta = map_iteration(theta, psf, img, alpha=alpha)

    return theta

def map_process_frame(frame, data, psf, flux, prev):
    img = cp.array(data[frame,...])
    psf = cp.array(psf)

    if not np.isscalar(flux):
        flux = flux[frame]

    beta = cp.zeros_like(prev)
    beta0 = 1.0 / _default_config["brightness"] / flux
    beta[prev != 0.0] = beta0

    alpha = prev * (img.sum() * beta0 / prev.sum()) + _default_config["uninformative_a"]
    #alpha[alpha == 0.0] = _default_config["sparse_alpha"]

    theta = cp.ones_like(prev)
    for i in range(_default_config["map_iterations"]):
        theta = map_iteration(theta, psf, img, alpha=alpha, beta=beta)

    return theta
