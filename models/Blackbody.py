#!/usr/bin/env python3
import csv
import numpy as np
from numpy import exp
from scipy.constants import c, h, k, pi

import sys
sys.path.append('../backend')

from Model import Model


class Blackbody(Model):
  def __init__(self, T=50, lambd=(8e-6, 14e-6), phi=(0,0), area=17e-6**2, omega=0.75):
    super().__init__(input_tuple=None, output_tuple={"P"})
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


  def store(self, prefix, args, input_data, output_data):
    with open(prefix + str(args) + '.csv', 'w') as file:
      csvwriter = csv.writer(file)
      csvwriter.writerow('Power')
      csvwriter.writerow([output_data["P"]])

  def load(self, prefix, args, input_data):
    try:
      with open(prefix + str(args) + '.csv', 'r') as file:
        csvreader = csv.reader(file)
        header = next(csvreader)
        data   = next(csvreader)
        return {"P": float(data[0])}
    except:
      return None

  def get_parameter_id_str(self, args, input_data):
    return str(args)
