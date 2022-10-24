#!/usr/bin/env python3
import csv
import numpy as np
from numpy import exp
from scipy.constants import c, h, k, pi

import sys
sys.path.append('../backend')

from Model import Model
import params

class Blackbody(Model):
  """
  Models IR camera sensor observing a black body radiator. Given the
  temperature or temperature range of the black body, calculates the
  IR power incident on a single pixel in the center of the imager's
  focal plane.
  
  Input data
  ----------
  None.
  
  Output data
  -----------
  P :
      IR power observed by the center pixel

  Model argument
  --------------
      Black body temperature in Kelvin.
   
  Initializer parameters
  ----------------------
  T :
      Black body temperature in Kelvin.
  
  lambd :
      Lower and upper bounds of the wavelength range of interest.
  
  phi :
      Angle of incidence.
  
  area :
      Sensitive area of a receiver's pixel.
  
  omega :
      Projected solid angle.
  """
  
  def __init__(self, T=params.T, lambd=params.lambd, phi=params.phi, area=params.area, omega=params.omega):
    super().__init__(input_tuple=None, output_tuple={"P": (1,)})
    self.T     = T
    self.lambd_lower, self.lambd_upper = lambd
    self.phi_r, self.phi_s = phi
    self.area  = area
    self.omega = omega

  def process(self, input_data=None, args=None):

    if args:
      T = args
    else:
      T = self.T

    L = 0.0
    it = np.arange(1, 101, 1)
    x1 = (h*c)/(k*T*self.lambd_lower)
    x2 = (h*c)/(k*T*self.lambd_upper)

    '''Integration of Planks radiation function in wave length of interest band
       "BLACKBODY RADIATION FUNCTION" Chang, Rhee 1984'''
    for n in it:
      B1 = (2 * k**4 * T**4)/(h**3 * c**2) * exp(-n*x1) * (x1**3 / n + (3 * x1**2)/n**2 + (6 * x1)/n**3 + 6/n**4)
      B2 = (2 * k**4 * T**4)/(h**3 * c**2) * exp(-n*x2) * (x2**3 / n + (3 * x2**2)/n**2 + (6 * x2)/n**3 + 6/n**4)
      L += (B2-B1)

    '''Calculating IR power that impignes on one pixel sensetive area, if it is
       located in the center of sensor'''
    P = L*np.cos(self.phi_s)*self.area*np.cos(self.phi_r)*self.omega
    return {"P":P}

  def get_parameter_id_str(self, args, input_data):
    return str(args)

