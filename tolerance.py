import sys_param as sp
import numpy as np
from scipy.constants import k
import argparse
import os

# System parameters defined in "sys_param.py"
T_s = sp.T_sa
R_tai = sp.R_ta_i
gi = sp.g_ini
ci = sp.c_ini
alpha = sp.alpha

Ea = sp.Ea
pix_v = sp.pix_v
pix_h = sp.pix_h

R_tol_d = 0.0005
g_tol_d = 0.0005
c_tol_d = 0.0005

parser = argparse.ArgumentParser(
    description='''
    Generate 480x640 arrays for physical parameters of MBA:
    "g" - thermal conductivity
    "c" - thermal capacity
    "tau" - thermal time constant
    "R_ta" - microbolometers resistance at ambient temperature
    Values can generated in user defined or default tolerance range.
    '''
    )

parser.add_argument('R_tol', type=float, 
  help='Standard deviation for normal distribution of MBA resistance \
  values at ambient temperature 300 K as coefficient for nominal value')

parser.add_argument('g_tol', type=float,
  help='Standard deviation for normal distribution of "g" as coefficient \
  for nominal value')

parser.add_argument('c_tol', type=float,
  help='Standard deviation for normal distribution of "c" as coefficient \
  for  nominal value')

args = parser.parse_args()

fdir = 'tolerance_data'
try:
  os.mkdir(fdir)
except OSError:
  print('\nDirectory "%s" already exist\n' % fdir)
else:
  print('\nSuccessfully created the directory "%s" \n' % fdir)

R_scale = R_tai*args.R_tol # Default stardard deviation for "R_ta" values
g_scale = gi*args.g_tol   # Default standard deviation for "g" values
c_scale = ci*args.c_tol   # Default standard deviation for "c" values


def generate_arrays(R_tol, g_tol, c_tol):
  global R_tai, gi, ci, pix_v, pix_h
  print("Koeficients used:")
  print("R_tol: ", R_tol)
  print("g_tol: ", g_tol)
  print("c_tol: ", c_tol)
  Rta = np.random.normal(loc=R_tai, scale=R_tol, size=(pix_v,pix_h))
  g = np.random.normal(loc=gi, scale=g_tol, size=(pix_v,pix_h))
  c = np.random.normal(loc=ci, scale=c_tol, size=(pix_v,pix_h))
    
  R0 = np.zeros((pix_v, pix_h))
  tau = np.zeros((pix_v, pix_h))
  row = np.arange(0,pix_v,1)
  column = np.arange(0,pix_h,1)
  for r in row:
    for col in column:
      R0[r][col] = Rta[r][col] / (np.exp(Ea/(k*T_s)))
      tau[r][col] = c[r][col] / g[r][col]
  print("\nSaving data:")
  np.savetxt('tolerance_data/Rta_tolerance.txt', Rta)
  np.savetxt('tolerance_data/g_tolerance.txt', g)
  np.savetxt('tolerance_data/c_tolerance.txt', c)
  np.savetxt('tolerance_data/R0_tolerance.txt', R0)
  np.savetxt('tolerance_data/tau_tolerance.txt', tau)
  print("Done.")
generate_arrays(R_scale, g_scale, c_scale)
