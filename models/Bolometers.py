#!/usr/bin/env python3
import os
import numpy as np
from numpy import exp
from PIL import Image
from scipy.constants import c, h, k, pi

import sys
sys.path.append('../backend')
from Model import Model

class Bolometers(Model):
  def __init__(self, Tcam=30, size_active=(320,240), size_boundary=(2,2,2,2),
               size_blind=(1,1,1,1), visualize=False,
               lambd=(8e-6, 14e-6), phi=(0,0), area=17e-6**2, omega=0.75,
               R_ambient_med=1e6,  R_ambient_dev=1e-5,
               G_thermal_med=1e-7, G_thermal_dev=1e-5,
               C_thermal_med=5e-9, C_thermal_dev=1e-5,
               T_ambient=300, TCR=-0.03):

    self.Tcam            = Tcam
    self.size_active_h   = size_active[0]
    self.size_active_v   = size_active[1]
    self.size_boundary_t = size_boundary[0]
    self.size_boundary_b = size_boundary[1]
    self.size_boundary_l = size_boundary[2]
    self.size_boundary_r = size_boundary[3]
    self.size_blind_t    = size_blind[0]
    self.size_blind_b    = size_blind[1]
    self.size_blind_l    = size_blind[2]
    self.size_blind_r    = size_blind[3]

    self.lambd_lower = lambd[0]
    self.lambd_upper = lambd[1]
    self.phi_r = phi[0]
    self.phi_s = phi[1]
    self.area  = area
    self.omega = omega

    self.size_total_h = self.size_active_h + self.size_boundary_l + self.size_boundary_r \
                      + self.size_blind_l + self.size_blind_r
    self.size_total_v = self.size_active_v + self.size_boundary_t + self.size_boundary_b \
                      + self.size_blind_t + self.size_blind_b

    super().__init__(input_tuple ={"P_distribution" : size_active},
                     output_tuple={"P_total"   : (self.size_total_h, self.size_total_v),
                                   "R_ambient" : (self.size_total_h, self.size_total_v),
                                   "G_thermal" : (self.size_total_h, self.size_total_v),
                                   "C_thermal" : (self.size_total_h, self.size_total_v),
                                   "R0"        : (self.size_total_h, self.size_total_v),
                                   "tau"       : (self.size_total_h, self.size_total_v)},
                     visualize=visualize)

    self.R_ambient_med = R_ambient_med
    self.R_ambient_dev = R_ambient_dev
    self.G_thermal_med = G_thermal_med
    self.G_thermal_dev = G_thermal_dev
    self.C_thermal_med = C_thermal_med
    self.C_thermal_dev = C_thermal_dev
    self.T_ambient     = T_ambient
    self.E_activation  = -(TCR * k *self.T_ambient**2)
    print(self.E_activation)



    # Model is parametirized using camera's temperatures, store it into arguments
    # for caching mechanism
    super().set_args_list([Tcam])


  def _get_temperature_power_component(self, T):
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
    return L*np.cos(self.phi_s)*self.area*np.cos(self.phi_r)*self.omega

  def _get_physical_parameters(self):
    size = (self.size_total_v, self.size_total_h)
    rng = np.random.default_rng(None)
    R_ambient = rng.normal(loc=self.R_ambient_med, scale=self.R_ambient_dev, size=size)
    G_thermal = rng.normal(loc=self.G_thermal_med, scale=self.G_thermal_dev, size=size)
    C_thermal = rng.normal(loc=self.C_thermal_med, scale=self.C_thermal_dev, size=size)

    tau = np.zeros((self.size_total_v, self.size_total_h))
    R0  = R_ambient / np.exp(self.E_activation/(k*self.T_ambient))

    return R_ambient, G_thermal, C_thermal, R0, tau

  def process(self, input_data=None, args=None):
    ''' Camera temperatures as a parameter '''
    if args:
      Tcam = args
    else:
      Tcam = self.Tcam

    ''' Calculate inherent camera temperature (just black-body based
        integration) '''
    P_temperature = self._get_temperature_power_component(Tcam)
    P_active      = input_data["P_distribution"]

    ''' Construct array for output pixels '''
    P_pixels = np.zeros((self.size_total_v, self.size_total_h))

    ''' Fill active area with distributed power '''
    active_start_h = self.size_boundary_l + self.size_blind_l
    active_stop_h  = self.size_total_h-1-1 - self.size_boundary_r + self.size_blind_r
    active_start_v = self.size_boundary_t + self.size_blind_t
    active_stop_v  = self.size_total_v-1-1 - self.size_boundary_b + self.size_blind_b
    P_pixels[active_start_v:active_stop_v,active_start_h:active_stop_h] = P_active
    P_total = P_pixels + P_temperature

    ''' Retrieve vital bolometer characteristics '''
    R_ambient, G_thermal, C_thermal, R0, tau = self._get_physical_parameters()

    return {
      "P_total"   : P_total,
      "R_ambient" : R_ambient,
      "G_thermal" : G_thermal,
      "C_thermal" : C_thermal,
      "R0"        : R0,
      "tau"       : tau}

  def store(self, prefix, args, input_data, output_data):
    # TODO: add hashsum check
    for key in output_data:
      np.savetxt(prefix + key, output_data[key])

  def load(self, prefix, args, input_data):
    try:
      return {
        "P_total"   : np.loadtxt(prefix + "P_total.txt"),
        "R_ambient" : np.loadtxt(prefix + "R_ambient.txt"),
        "G_thermal" : np.loadtxt(prefix + "G_thermal.txt"),
        "C_thermal" : np.loadtxt(prefix + "C_thermal.txt"),
        "R0"        : np.loadtxt(prefix + "R0.txt"),
        "tau"       : np.loadtxt(prefix + "tau.txt")}
    except:
      return None

  def store_display(self, prefix, args, input_data, output_data, cached):
    fname = prefix + '.png'
    if not os.path.exists(fname) or not cached:
      output_normalized = output_data["P_total"] * (255.0/output_data["P_total"].max())
      image_data = output_normalized.astype(np.uint8)
      image = Image.fromarray(image_data)
      image.save(fname)

  def get_parameter_id_str(self, args, input_data):
    return str(args)
