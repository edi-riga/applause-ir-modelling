import numpy as np
from scipy.constants import k

from . import sys_param as sp


def generate_arrays(R_tol=sp.R_tol, g_tol=sp.g_tol, c_tol=sp.c_tol, size=(sp.pix_v_all, sp.pix_h_all), R_tai = sp.R_ta_i, gi=sp.g_ini, ci=sp.c_ini, Ea=sp.Ea, T_s=sp.T_sa, seed=None):
    """
    Generate arrays of microbolometer physical parameters
    
    Parameters
    ----------
    R_tol, g_tol, c_tol: float, optional
        Standard deviation coefficients for resistance at ambient 
        temperature, thermal conductivity and thermal capacity.
        Standard deviation values will be calculated as product
        of these coefficients and the nominal values.
    
    size: (float, float), optional
        Height and width of the microbolometer array.
    
    R_tai, gi, ci: float, optional
        Nominal values for resistance at ambient temperature,
        thermal conductivity and thermal capacity.
    
    Ea: float, optional
        Activation energy.
    
    T_s: float, optional
        Sample time.
    
    seed: optional
        Seed for the random number generator. For more information,
        see the numpy.random.default_rng documentation.
    
    Returns
    -------
    Rta, g, c, R0, tau:
        Arrays of microbolometer resistance at ambient temperature,
        thermal conductivity, thermal capacity, ? and thermal time constant.
    """
    
    pix_v_all, pix_h_all = size
    
    R_scale = R_tai*R_tol  # Stardard deviation for "R_ta" values
    g_scale = gi*g_tol     # Standard deviation for "g" values
    c_scale = ci*c_tol     # Standard deviation for "c" values
    
    print('Standard deviation coefficients used:')
    print(f'R_tol: {R_tol}')
    print(f'g_tol: {g_tol}')
    print(f'c_tol: {c_tol}')
    
    rng = np.random.default_rng(seed)
    
    if seed is not None:
        print(f'Seed: {seed}')
    else:
        print('Using random seed')
    
    Rta = rng.normal(loc=R_tai, scale=R_scale, size=size)
    g = rng.normal(loc=gi, scale=g_scale, size=size)
    c = rng.normal(loc=ci, scale=c_scale, size=size)
    R0 = np.zeros((pix_v_all, pix_h_all))
    tau = np.zeros((pix_v_all, pix_h_all))
    row = np.arange(pix_v_all)
    column = np.arange(pix_h_all)
    R0 = Rta / np.exp(Ea/(k*T_s))
    tau = c / g
    
    return Rta, g, c, R0, tau

