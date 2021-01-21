import sys
import numpy as np
from numpy import exp
from scipy.constants import c, h, k, pi

from . import sys_param as sp


def integration(T, lambd=(sp.lambd1, sp.lambd2), Phi=(sp.Phi_r, sp.Phi_s), A_sens=sp.A_sens, Omega=sp.Omega):
  temper_it = np.arange(T.size)
  
  lambd1, lambd2 = lambd
  Phi_r, Phi_s = Phi
  
  # Empty array for radiance integral results
  L = np.zeros(T.size)
  # Number of terms in the intergation row
  it = np.arange(1, 101, 1)

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
  
  ''' Calculating IR power that impignes on
      one pixel sensetive area, if it is 
      located in the center of sensor'''
  P = L*np.cos(Phi_s)*A_sens*np.cos(Phi_r)*Omega
  return P


# Define function of IR power distribution over sensor area.
def power_distribution_over_sens_area(P, size=(sp.pix_v, sp.pix_h), fl=sp.fl, pitch = sp.pitch):
  '''
  size = (pix_v, pix_h)
  fl = Focal length
  
  '''
  
  pix_v, pix_h = size
  
  # Define array for saving distribution 
  # factor results for each pixel
  distrib_fact = np.ones(size)
  
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

