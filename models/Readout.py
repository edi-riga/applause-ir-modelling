#!/usr/bin/env python3
import os
import numpy as np
from numpy import exp
from scipy.optimize import fsolve
from PIL import Image
from scipy.constants import c, h, k, pi

import sys
sys.path.append('../backend')
from Model import Model

class Readout(Model):
  def __init__(self, size_active=(320,240), size_boundary=(2,2,2,2), size_blind=(1,1,1,1),
               I_bias=50e-6, E_act=3.7277523e-20, T_amb=300, t_int=10e-3,
               V_max=3.2,
               visualize=False):

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

    self.size_total_h = self.size_active_h + self.size_boundary_l + self.size_boundary_r \
                      + self.size_blind_l + self.size_blind_r
    self.size_total_v = self.size_active_v + self.size_boundary_t + self.size_boundary_b \
                      + self.size_blind_t + self.size_blind_b

    self.I_bias = I_bias
    self.E_act  = E_act
    self.T_amb  = T_amb
    self.t_int  = t_int
    self.V_max  = V_max

    super().__init__(
      input_tuple={
        "P_total"   : (self.size_total_h, self.size_total_v),
        "R_ambient" : (self.size_total_h, self.size_total_v),
        "G_thermal" : (self.size_total_h, self.size_total_v),
        "C_thermal" : (self.size_total_h, self.size_total_v),
        "R0"        : (self.size_total_h, self.size_total_v),
        "tau"       : (self.size_total_h, self.size_total_v)},
      output_tuple = {"DAC": (self.size_active_h, self.size_active_v)},
      visualize=visualize)

    self.mask_active   = np.zeros((self.size_total_v, self.size_total_h)).astype(bool)
    self.mask_boundary = np.zeros((self.size_total_v, self.size_total_h)).astype(bool)
    self.mask_blind    = np.ones((self.size_total_v, self.size_total_h)).astype(bool)

    self.mask_active[
      self.size_boundary_t+self.size_blind_t
      : self.size_total_v-self.size_boundary_b-self.size_blind_b,
      self.size_boundary_l+self.size_blind_l
      : self.size_total_h-self.size_boundary_r-self.size_blind_r] = True

    self.mask_blind[0:self.size_blind_t, :]                                   = True
    self.mask_blind[self.size_total_v-self.size_blind_b:self.size_total_v, :] = True
    self.mask_blind[:, 0:self.size_blind_l]                                   = True
    self.mask_blind[:, self.size_total_h-self.size_blind_r:self.size_total_v] = True

    self.mask_boundary = np.logical_not(np.logical_or(self.mask_active,self.mask_blind))

    # RESISTORS ON OP AMP INPUTS (Very model-specific)
    self.R1 = 200e6
    self.R2 = 10e3
    self.R3 = 400e6
    # GAIN CAPACITOR OF INTEGRATOR
    self.C  = 0.2e-12




  def process(self, input_data=None, args=None):

    T_amb = args['T_amb'] if args and args['T_amb'] else self.T_amb
    t_int = args['t_int'] if args and args['t_int'] else self.t_int

    Q   = input_data["P_total"]
    R0  = input_data["R0"]
    G   = input_data["G_thermal"]
    tau = input_data["tau"]

    V_bol   = np.zeros((self.size_total_v, self.size_total_h))
    V_bol_h = np.zeros((self.size_active_v, self.size_active_h))
    V_bol_v = np.zeros((self.size_active_v, self.size_active_h))

    '''Calculate output voltage of each pixel at the given bias current'''
    print("\n")
    for r, c in np.argwhere(np.logical_or(self.mask_active, self.mask_boundary)):
      print("\rProcessing pixel hold voltage for row: " + str(r), end="")
      V_bol[r,c] = fsolve(
        lambda Va: Va - self.I_bias*R0[r, c]
          * exp(self.E_act
             / (k * T_amb
               + k*((self.I_bias*Va + Q[r,c]) / G[r, c])
               * (1 + (tau[r, c] / t_int)
                  * (exp(-t_int/tau[r, c]) - 1)))),
        0)

    print("\n")
    for r, c in np.argwhere(self.mask_blind):
      print("\rProcessing pixel hold voltage for row: " + str(r), end="")
      V_bol[r,c] = fsolve(
        lambda Va: Va - self.I_bias*R0[r, c]
          * exp(self.E_act
             / (k * T_amb
               + k*((self.I_bias*Va + 0) / G[r, c])
               * (1 + (tau[r, c] / t_int)
                  * (exp(-t_int/tau[r, c]) - 1)))),
        0)
    print("\n")

    '''Recalculate output voltage using theoretical temperature correction
    that utilizes analog (voltage-based) subtraction, calculate both
    vertical and horizontal bolometer correction effects'''
    for r in range(self.size_active_v):
      for c in range(self.size_active_h):
        blind_t = V_bol[r,0:self.size_blind_t]
        blind_b = V_bol[r,self.size_total_v-self.size_blind_b:self.size_total_v]
        blind_l = V_bol[0:self.size_blind_l, c]
        blind_r = V_bol[self.size_total_v-self.size_blind_r:self.size_total_v, c]
        blind_average_v = np.average([blind_t, blind_b])
        blind_average_h = np.average([blind_l, blind_r])

        V_bol_v[r,c] = (1/(self.R1*self.C)) \
                       * ((self.R3 /(self.R2+self.R3)) \
                       * blind_average_v \
                       - V_bol[r+self.size_blind_t + self.size_boundary_t,
                               c+self.size_blind_l + self.size_boundary_l] ) \
                       + (self.R3 /(self.R2+self.R3) * blind_average_v)

        V_bol_h[r,c] = (1/(self.R1*self.C)) \
                       * ((self.R3 /(self.R2+self.R3)) \
                       * blind_average_h \
                       - V_bol[r+self.size_blind_t + self.size_boundary_t,
                               c+self.size_blind_l + self.size_boundary_l] ) \
                       + (self.R3 /(self.R2+self.R3) * blind_average_h)

        V_bol_v[r,c] = V_bol_v[r,c] if V_bol_v[r,c] > 0 else 0
        V_bol_v[r,c] = V_bol_v[r,c] if V_bol_v[r,c] <= self.V_max else self.V_max
        V_bol_h[r,c] = V_bol_h[r,c] if V_bol_v[r,c] > 0 else 0
        V_bol_h[r,c] = V_bol_h[r,c] if V_bol_v[r,c] <= self.V_max else self.V_max

    return {
      "V_bol"   : V_bol,
      "V_bol_h" : V_bol_h,
      "V_bol_v" : V_bol_v}


  def store(self, prefix, args, input_data, output_data):
    # TODO: add hashsum check\
    for key in output_data:
      print("Storing:" + key)
      np.savetxt(prefix + key, output_data[key])

  def load(self, prefix, args, input_data):
    print(prefix + "V_bol")
    print(prefix + "V_bol_h")
    print(prefix + "V_bol_v")
    try:
      return {"V_bol"   : np.loadtxt(prefix + "V_bol"),
              "V_bol_h" : np.loadtxt(prefix + "V_bol_h"),
              "V_bol_v" : np.loadtxt(prefix + "V_bol_v")}
    except:
      return None

  def store_display(self, prefix, args, input_data, output_data, cached):
    for key in output_data:
      print("Displaying" + key)
      fname = prefix + key + '.png'
      if not os.path.exists(fname) or not cached:
        output_normalized = output_data[key] * (255.0/output_data[key].max())
        image_data = output_normalized.astype(np.uint8)
        image = Image.fromarray(image_data)
        image.save(fname)

  def get_parameter_id_str(self, args, input_data):
    T_amb = args['T_amb'] if args and args['T_amb'] else self.T_amb
    t_int = args['t_int'] if args and args['t_int'] else self.t_int
    print(T_amb)
    print(t_int)
    return str(T_amb) + "_" + str(t_int)
