#!/usr/bin/python3
import os
import numpy as np
from scipy.constants import k

from . import sys_param as sp

def generate_arrays(R_tol=sp.R_tol, g_tol=sp.g_tol, c_tol=sp.c_tol, size=(sp.pix_v_all, sp.pix_h_all), R_tai = sp.R_ta_i, gi=sp.g_ini, ci=sp.c_ini, Ea=sp.Ea, T_s=sp.T_sa):
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
    
    
    print("Coefficients used:")
    print("R_tol: ", R_tol)
    print("g_tol: ", g_tol)
    print("c_tol: ", c_tol)
    Rta = np.random.normal(loc=R_tai, scale=R_scale, size=size)
    g = np.random.normal(loc=gi, scale=g_scale, size=size)
    c = np.random.normal(loc=ci, scale=c_scale, size=size)
    R0 = np.zeros((pix_v_all, pix_h_all))
    tau = np.zeros((pix_v_all, pix_h_all))
    row = np.arange(pix_v_all)
    column = np.arange(pix_h_all)
    for r in row:
        for col in column:
            R0[r][col] = Rta[r][col] / (np.exp(Ea/(k*T_s)))
            tau[r][col] = c[r][col] / g[r][col]
    
    return Rta, g, c, R0, tau


def save_data(fdir, Rta, g, c, R0, tau):
  try:
      os.mkdir(fdir)
  except OSError:
      pass
  print("\nSaving data:")
  np.savetxt(f'{fdir}/Rta_tolerance.txt', Rta)
  np.savetxt(f'{fdir}/g_tolerance.txt', g)
  np.savetxt(f'{fdir}/c_tolerance.txt', c)
  np.savetxt(f'{fdir}/R0_tolerance.txt', R0)
  np.savetxt(f'{fdir}/tau_tolerance.txt', tau)
  print("Done.")


def load_data(fdir):
  Rta = np.loadtxt(f'{fdir}/Rta_tolerance.txt')
  g = np.loadtxt(f'{fdir}/g_tolerance.txt')
  c = np.loadtxt(f'{fdir}/c_tolerance.txt')
  R0 = np.loadtxt(f'{fdir}/R0_tolerance.txt')
  tau = np.loadtxt(f'{fdir}/tau_tolerance.txt')
  return Rta, g, c, R0, tau
