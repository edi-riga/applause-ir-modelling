```python
import sys
import numpy as np
from numpy import exp
from scipy.constants import c, h, k, pi
import sys_param as sp
import matplotlib.pyplot as plt
import argparse
import os


# Initializing argparse
parser = argparse.ArgumentParser(
    description='''
    Calculates IR power "P" distibution over sensor area "Pa"
    emmited by the blackbody at given temperature "T".
    '''
    )
    
#Adding the arguments
group = parser.add_mutually_exclusive_group()
group.add_argument('-d', '--distribution', action="store_true", 
                   help='Calculate Pa(T) @ each T and save the results to .txt files')
group.add_argument('-g', '--graph', action="store_true",
                   help='Calculate and plot P(T). Default option. Explain!', default=1)
parser.add_argument('T_start', type=float, help="Initial temperature of blackbody", default=300)
parser.add_argument('T_stop',  type=float, help="End temperature of blackbody")
parser.add_argument('T_step',  type=float, help="Step of temperature increase")
parser.add_argument('-r', '--run', action="store_true",
                    help='Used only if "blackbody.py" have been called from "frame_gen.py"', default=0 )

if len(sys.argv) == 1:
  parser.print_help(sys.stderr)
  sys.exit(1)
args = parser.parse_args()
```
Importing all necessary and initializing argparse

----


```python
def names():
  # Names for output files and directories
  global fdir, pattern1, pattern2, pattern3, pattern4, fname1, fname2
  fdir = 'data_files'
  pattern1 = 'pto1pix'
  pattern2 = 'distribution'
  pattern3 = '_BB_TempRange_'
  pattern4 = str(args.T_start) + '_' + str(args.T_stop) + '_' + str(args.T_step) + '.txt'
  fname1 = fdir + '/' + pattern1 + pattern3 + pattern4
  fname2 = fdir + '/' + pattern2 + pattern3 + pattern4
```
String used for names of output files and directories

----

```python
def collect_data():
  # System parameters defined in "sys_param.py"
  global Omega, Ar, A_pix, A_sens
  Omega = sp.Omega
  Ar = sp.Ar
  A_pix = sp.A_pix
  A_sens = sp.A_sens

  global Phi_r, Phi_s, tau
  Phi_r = sp.Phi_r
  Phi_s = sp.Phi_r
  tau = sp.tau

  global pix_h, pix_v, pitch, fl, lambd1, lambd2
  pix_h = sp.pix_h
  pix_v = sp.pix_v
  pitch = sp.pitch
  fl = sp.fl
  lambd1 = sp.lambd1
  lambd2 = sp.lambd2
```
Collecting data from "sys_param(.py)" module. 
For detailed information see "sys_param_explonation.md"

----

Performing in-band integration of Planck radiation function, 
using method described in sources:
> 1. "Inegration of Planck blackbody radiation function." W.K.Widger, M.P. Woodall

> 2. "Blackbody Radiation Function." S.L. Chang, K.T. Rhee

 - Defining list of blackbody temperatures and iterative row:
```python
def integration():
  # Array of blackboby temperatures
  global T, temper_it, L, it
  T = np.arange(args.T_start, args.T_stop+args.T_step, args.T_step)
  temper_it = np.arange(T.size)
```

 - Defining empty array to store calculation results and iterative row length:
```python
  # Empty array for radiance integral results
  L = np.zeros(T.size)
  # Number of terms in the intergation row
  it = np.arange(1, 101, 1)
```

 - Performing calculations: 
```python
  ''' Integration of Planks radiation function
      in wave length of interest band
      "BLACKBODY RADIATION FUNCTION" Chang, Rhee 1984'''
  for t in temper_it:
    x1 = (h*c)/(k*T[t]*lambd1)
    x2 = (h*c)/(k*T[t]*lambd2)
    for n in it:
      B1 = (2 * k**4 * T[t]**4)/(h**3 * c**2) * exp(-n*x1) * (x1**3 / n + (3 * x1**2)/n**2 + (6 * x1)/n**3 + 6/n**4)
      B2 = (2 * k**4 * T[t]**4)/(h**3 * c**2) * exp(-n*x2) * (x2**3 / n + (3 * x2**2)/n**2 + (6 * x2)/n**3 + 6/n**4)
      L[t] += (B2-B1)
```

 - Calculating IR power that impignes on one pixel sensetive area, if it is located in the center of sensor
```python
  global P
  ''' Calculating IR power that impignes on
      one pixel sensetive area, if it is 
      located in the center of sensor'''
  P = L*np.cos(Phi_s)*A_sens*np.cos(Phi_r)*Omega
```


# Argument "--distribution"
# Define function of IR power distribution over sensor area.
def power_distribution_over_sens_area():
  global fl, pow_distrib, P
  
  # Define array for saving distribution 
  # factor results for each pixel
  distrib_fact = np.ones((pix_v, pix_h))
  
  # Define variables for iterations
  # over one quadrant of the sensor
  row_half = pix_v/2
  col_half = pix_h/2
  half_pitch = pitch/2
  row = np.arange(0, int(row_half), 1)
  col = np.arange(0, int(col_half), 1)
  
  # Calculating IR power distribution factor
  for r in row:
    row1 = int(row_half - 1 - r)
    row2 = int(row_half + r)
    
    for co in col:
        a = np.sqrt( (half_pitch + r * pitch) ** 2 + (half_pitch + co * pitch) ** 2 )
        b = np.sqrt(a ** 2 + fl ** 2)
        fact = (fl/b)**4
        col1 = int(col_half + co)
        col2 = int(col_half - 1 - co)
        
        distrib_fact[ row1 ][ col1 ] = fact # 1st quadrant
        distrib_fact[ row2 ][ col1 ] = fact # 2nd quadrant
        distrib_fact[ row2 ][ col2 ] = fact # 3rd quadrant
        distrib_fact[ row1 ][ col2 ] = fact # 4th quadrant
  
  # Define iterative rows for IR power distribution calculation
  row_pd = np.arange(0, pix_v, 1)
  col_pd = np.arange(0, pix_h, 1)
  pow_ar = np.arange(0, P.size, 1)
  
  # Define array for savin power distribution
  # calculated values
  pow_distrib = np.ones((P.size, pix_v, pix_h))
  
  # Calculating power IR power distribution over sensor area
  for p in pow_ar:
    for r in row_pd:
        for co in col_pd:
            pow_distrib[p][r][co] = distrib_fact[r][co]*P[p]
  return pow_distrib


# Argument "--graph"
def plot_result():
  plt.plot(T, P, 'bx')
  plt.xlabel('Black body temperature [K]')
  plt.ylabel('IR power [W], impigned on one pixel sensitive area, \n located in the middle of the sensor')
  plt.grid(True)
  plt.show()


def save_data():
  if args.run:
    np.savetxt('buf_of_powers.txt', P)

    with open('buf_of_data.txt','w') as outfile:
      for data in pow_distrib:
        np.savetxt(outfile, data)
    print("Done.\n")
  else:
    #global fdir, fname1, fname2, P, pow_distrib
    try:
      os.mkdir(fdir)
    except OSError:
      print('\nDirectory "%s" already exist\n' % fdir)
    else:
      print('\nSuccessfully created the directory "%s" \n' % fdir)

    print("\nSaving data")
    # Saving IR power values impigned on one pixel
    np.savetxt(fname1, P)

    # Saving IR distribution over sensor area
    with open(fname2, 'w') as outfile:
      for data in pow_distrib:
        np.savetxt(outfile, data)
    print("Done.\n")


if __name__ == '__main__':
  names()
  collect_data()
  integration()
  if args.distribution:
    # Argument "--distribution"
    print("Blackbody temperature is continiously growing")
    print("from {} K to {} K with {} K step".format(args.T_start, args.T_stop, args.T_step))
    print("\nCalculating the distribution of IR power")
    print("over sensor area @ blackbody temperatures:")
    print(T)
    power_distribution_over_sens_area()
    save_data()
    
  elif args.graph:
    # Argument "--graph"
    plot_result()
    print("The grafical results will be showed")
