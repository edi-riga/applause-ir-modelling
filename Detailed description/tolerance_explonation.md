#### Import necessary modules and initialize argument parser
```python
import sys
import sys_param as sp
import numpy as np
from scipy.constants import k
import argparse
import os


parser = argparse.ArgumentParser(
    description='''
    Generate 486x646 arrays for physical parameters of MBA:
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


if len(sys.argv) == 1:
  parser.print_help(sys.stderr)
  sys.exit(1)
args = parser.parse_args()
```

----

#### Collect necessary data from "sys_param(.py)" module
```python
def collect_data():
  # System parameters defined in "sys_param.py"
  global T_s, R_tai, gi, ci, alpha
  T_s = sp.T_sa
  R_tai = sp.R_ta_i
  gi = sp.g_ini
  ci = sp.c_ini
  alpha = sp.alpha
  global Ea, pix_v, pix_h, pix_v_all, pix_h_all
  Ea = sp.Ea
  pix_v = sp.pix_v
  pix_h = sp.pix_h
  pix_v_all = sp.pix_v_all
  pix_h_all = sp.pix_h_all
```

----

#### Create directory where to save tolerance data
```python
def make_dir():
  global fdir
  fdir = 'tolerance_data'
  try:
    os.mkdir(fdir)
  except OSError:
    print('\nDirectory "%s" already exist\n' % fdir)
  else:
    print('\nSuccessfully created the directory "%s" \n' % fdir)
```

----

#### Define standard deviations for each parameter using argparse arguments.

The main assumption is that variation in parameter values has normal distribution.
Standard deviation for each parameter is set as multiplication of nominal value and
corresponding positional argparse argument.
Thus argparse arguments are coefficients for nominal values of physical parameters.

```python
def standard_deviation():
  global R_scale, g_scale, c_scale
  R_scale = R_tai*args.R_tol  # Stardard deviation for "R_ta" values
  g_scale = gi*args.g_tol     # Standard deviation for "g" values
  c_scale = ci*args.c_tol     # Standard deviation for "c" values
```

----

#### Create arrays of physical parameters using

Random generator is used to create arrays of physical parameters, variating in values.
The dimensions of arrays corresponds to sensor dimensions in pixels.
Once created - the tolerance data arrays are used in all the further calculations.
Thus - each simulated pixel has it's own values of physical parameters.
Every time running "tolerance(.py)" script, new arrays of tolerance data are created and stored.
Thus - the arrays of tolerance ensures the fixed pattern noise, or non-uniformity.

```python
def generate_arrays(R_toler, g_toler, c_toler):
  global R_tai, gi, ci, pix_v, pix_h, g, c, Rta
  print("Coefficients used:")
  print("R_tol: ", R_toler)
  print("g_tol: ", g_toler)
  print("c_tol: ", c_toler)
  Rta = np.random.normal(loc=R_tai, scale=R_toler, size=(pix_v_all,pix_h_all))
  g = np.random.normal(loc=gi, scale=g_toler, size=(pix_v_all,pix_h_all))
  c = np.random.normal(loc=ci, scale=c_toler, size=(pix_v_all,pix_h_all))
  R0 = np.zeros((pix_v_all, pix_h_all))
  tau = np.zeros((pix_v_all, pix_h_all))
  row = np.arange(pix_v_all)
  column = np.arange(pix_h_all)
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
```

----

#### Calling script directly from terminal

```python
if __name__ == '__main__':
  collect_data()
  make_dir()
  standard_deviation()
  generate_arrays(R_scale, g_scale, c_scale)
```