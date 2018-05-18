import numpy as np

# Type Conversions

def to_uint8(img):
    mx,mn = np.max(img),np.min(img)
    img = (img-mn)/(mx-mn)*255
    return np.uint8(img)

# Histograms

def make_power_level_func(gam):
    return lambda x:np.uint8(255*(255**-gam)*x**gam)

def log_levels_func(levels):
    return np.uint8(np.log(1+levels)/np.log(255)*255)

def multithreshold(im,n_levs,levels=-1, levels_func=lambda x: x, is_inv_im=False):
    new_im = im.copy()
    if is_inv_im:
        new_im = 255-new_im
    if levels==-1:
        levels = np.hstack((np.arange(0,255,256/n_levs),255))
    levels = levels_func(levels)
    for i, level in enumerate(levels[1:]):
        new_im[(new_im>levels[i]) & (new_im<=level)] = level
    new_im[new_im==0]=levels[1]
    return new_im

# Filtros

def filter_ideal_fcn(H,D,D0,params):
    H[D<=D0] = 1
    return H
def filter_butterworth_fcn(H,D,D0,params=2):
    n = params
    H[:] = 1/(1+np.power(D[:]/D0,2*n))
    return H
def filter_gaussian_fcn(H,D,D0,params=2):
    n = params
    H[:] = np.exp(-np.power(D[:]/D0,2)/(n))
    return H

def get_filter_for_image(rows,cols,D0,filter_fcn,params=0,off_rows = 0, off_cols = 0):
    H = np.zeros((rows,cols))
    D = np.zeros_like(H)
    u, v= np.arange(0,rows), np.arange(0,cols)
    U = u[:rows].reshape((rows,1))*np.ones((1,cols))
    V = np.ones((rows,1))*v[:cols].reshape((1,cols))
    D = np.sqrt(np.power(U-rows/2-off_rows,2)+np.power(V-cols/2-off_cols,2))
    H = filter_fcn(H,D,D0,params)
    return H 

def get_notch_filter(rows,cols,D0,filter_fcn,params,u,v):
    H1 = get_filter_for_image(rows,cols,D0,filter_fcn,params,off_rows=u,off_cols=v)
    if u==0 and v==0:
        return H1
    H2 = get_filter_for_image(rows,cols,D0,filter_fcn,params,off_rows=-u,off_cols=-v)
    return H1+H2

def make_spectrum(img):
    f_im = np.fft.fft2(img)
    f_im_shift = np.fft.fftshift(f_im)
    mag_spect = 20*np.log(np.abs(f_im_shift))    
    return mag_spect

def apply_frecuency_filter_to_image(img, H):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f)
    f_ishift = np.fft.ifftshift(fshift*H)
    img_back = np.fft.ifft2(f_ishift)
    img_back = np.abs(img_back)
    return img_back