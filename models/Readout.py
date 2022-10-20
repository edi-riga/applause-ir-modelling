#!/usr/bin/env python3
import os
import cv2
import numpy as np
from numpy import exp
from scipy.optimize import fsolve
from PIL import Image
from scipy.constants import c, h, k, pi

import sys
sys.path.append('../backend')
from Model import Model
import params

class Readout(Model):
  """
  Model of a readout circuit. Calculates the measured microbolometer
  voltage levels. Three different outputs are produced, two of which
  partially compensate the nonuniformity using a skimming row or column,
  the third output does not perform any additional compensation.
  
  Input data
  -----------
  P_total :
      Total observed IR power (scene + camera body) across the
      full sensor's area.
  
  R_ambient :
      Microbolometer ambient temperature resistance distribution.
  
  G_thermal :
      Microbolometer thermal conductance distribution.
  
  C_thermal :
      Microbolomeer thermal capacity distribution.
  
  R0 :
      
  
  tau :
      Microbolometer thermal time constant distribution.
  
  Output data
  -----------
  V_bol :
      Raw microbolometer voltages.
  
  V_bol_h :
      Microbolometer voltages with horizontal skimming.
  
  V_bol_v :
      Microbolometer voltages with vertical skimming.

  Initializer parameters
  ----------------------
  size_active :
      Resolution of the sensor's active area.
  
  size_boundary :
      Number of boundary pixels on each side.
  
  size_blind :
      Number of blind pixels on each side.
  
  I_bias :
      Microbolometer bias current.
  
  E_act :
      Microbolometer activation energy.
  
  T_amb :
      Ambient temperature.
  
  t_int :
      ADC integration time.
  
  V_max :
      ADC maximum output voltage.
  """
  def __init__(self, size_active=params.resolution, size_boundary=params.size_boundary, size_blind=params.size_blind,
               I_bias=params.I_bias, E_act=params.E_act, T_amb=params.T_ambient, t_int=params.t_int,
               V_max=params.V_max,
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
      # output_tuple = {"DAC": (self.size_active_h, self.size_active_v)},
      output_tuple = {
        "V_bol"     : (self.size_total_h, self.size_total_v),
        "V_bol_h"   : (self.size_total_h, self.size_total_v),
        "V_bol_v"   : (self.size_total_h, self.size_total_v),
      },
      visualize=visualize)

    self.mask_active   = np.zeros((self.size_total_v, self.size_total_h)).astype(bool)
    self.mask_boundary = np.zeros((self.size_total_v, self.size_total_h)).astype(bool)
    self.mask_blind    = np.zeros((self.size_total_v, self.size_total_h)).astype(bool)

    self.mask_active[
      self.size_boundary_t+self.size_blind_t
      : self.size_total_v-self.size_boundary_b-self.size_blind_b,
      self.size_boundary_l+self.size_blind_l
      : self.size_total_h-self.size_boundary_r-self.size_blind_r] = True

    self.mask_blind[0:self.size_blind_t, :]                                   = True
    self.mask_blind[self.size_total_v-self.size_blind_b:self.size_total_v, :] = True
    self.mask_blind[:, 0:self.size_blind_l]                                   = True
    self.mask_blind[:, self.size_total_h-self.size_blind_r:self.size_total_h] = True

    self.mask_boundary = np.logical_not(np.logical_or(self.mask_active,self.mask_blind))

    # RESISTORS ON OP AMP INPUTS (Very model-specific)
    self.R1 = params.R1
    self.R2 = params.R2
    self.R3 = params.R3
    # GAIN CAPACITOR OF INTEGRATOR
    self.C  = params.C




  def process(self, input_data=None, args=None):

    T_amb = args['T_amb'] if args and args['T_amb'] else self.T_amb
    t_int = args['t_int'] if args and args['t_int'] else self.t_int

    Q   = input_data["P_total"]
    R0  = input_data["R0"]
    G   = input_data["G_thermal"]
    tau = input_data["tau"]

    V_int      = np.zeros((self.size_total_v, self.size_total_h))
    V_int_skim = np.zeros((self.size_total_v, self.size_total_h))
    V_bol      = np.zeros((self.size_total_v, self.size_total_h))
    V_bol_h    = np.zeros((self.size_active_v, self.size_active_h))
    V_bol_v    = np.zeros((self.size_active_v, self.size_active_h))
    print(self.E_act)
    print(T_amb)
    print(self.I_bias)
    print(t_int)

    print(self.R1)
    print(self.R2)
    print(self.R3)
    print(self.C)

    '''Calculate output voltage of each pixel at the given bias current'''
    print("")
    #for r, c in np.argwhere(np.logical_or(self.mask_active, self.mask_boundary)):
    for r in range(self.size_total_v):
      for c in range(self.size_total_h):
        print("\rProcessing pixel hold voltage for active pixels, row: " + str(r), end="")
        V_int[r,c] = fsolve(
          lambda Va: Va - self.I_bias*R0[r, c]
            * exp(self.E_act
               / (k * T_amb
                 + k*((self.I_bias*Va + Q[r,c]) / G[r, c])
                 * (1 + (tau[r, c] / t_int)
                    * (exp(-t_int/tau[r, c]) - 1)))),
          0)

    print("")
    for r, c in np.argwhere(self.mask_blind):
      print("\rProcessing pixel hold voltage for blind pixels, row: " + str(r), end="")
      V_int_skim[r,c] = fsolve(
        lambda Vs: Vs - self.I_bias * R0[r, c]
          * exp(self.E_act
              / (k * (T_amb
                  + ((self.I_bias*Vs + 0)/G[r, c])
                  * (1 - exp(-self.t_int/tau[r, c]))))),
        0)

    print("")

    '''Recalculate output voltage using theoretical temperature correction
    that utilizes analog (voltage-based) subtraction, calculate both
    vertical and horizontal bolometer correction effects'''
    for r in range(self.size_active_v):
      for c in range(self.size_active_h):
        blind_int_t = V_int[0:self.size_blind_t, c]
        blind_int_b = V_int[self.size_total_v-self.size_blind_b:self.size_total_v, c]
        blind_int_l = V_int[r, 0:self.size_blind_l]
        blind_int_r = V_int[r, self.size_total_h-self.size_blind_r:self.size_total_h]
        blind_skim_t = V_int_skim[0:self.size_blind_t, c]
        blind_skim_b = V_int_skim[self.size_total_v-self.size_blind_b:self.size_total_v, c]
        blind_skim_l = V_int_skim[r, 0:self.size_blind_l]
        blind_skim_r = V_int_skim[r, self.size_total_h-self.size_blind_r:self.size_total_h]
        blind_average_int_v  = np.average([blind_int_t, blind_int_b])
        blind_average_int_h  = np.average([blind_int_l, blind_int_r])
        blind_average_skim_v = np.average([blind_skim_t, blind_skim_b])
        blind_average_skim_h = np.average([blind_skim_l, blind_skim_r])

        V_bol[r + self.size_blind_t + self.size_boundary_t,
              c + self.size_blind_l + self.size_boundary_l] = \
                       (1/(self.R1*self.C)) \
                       * ((self.R3 /(self.R2+self.R3)) \
                       * blind_average_int_v \
                       - V_int[r+self.size_blind_t + self.size_boundary_t,
                               c+self.size_blind_l + self.size_boundary_l] )
        V_bol_v[r,c] = (1/(self.R1*self.C)) \
                       * ((self.R3 /(self.R2+self.R3)) \
                       * blind_average_int_v \
                       - V_int[r+self.size_blind_t + self.size_boundary_t,
                               c+self.size_blind_l + self.size_boundary_l] ) \
                       + (self.R3 /(self.R2+self.R3) * blind_average_skim_v)

        V_bol_h[r,c] = (1/(self.R1*self.C)) \
                       * ((self.R3 /(self.R2+self.R3)) \
                       * blind_average_int_h \
                       - V_int[r+self.size_blind_t + self.size_boundary_t,
                               c+self.size_blind_l + self.size_boundary_l] )\
                       + (self.R3 /(self.R2+self.R3) * blind_average_skim_h)

        #V_bol_v[r,c] =  V_bol[r+self.size_blind_t + self.size_boundary_t,
        #                      c+self.size_blind_l + self.size_boundary_l] \
        #                 - (self.R3 /(self.R2+self.R3) * blind_average_v)

        #V_bol_h[r,c] = V_bol[r+self.size_blind_t + self.size_boundary_t,
        #                     c+self.size_blind_l + self.size_boundary_l] \
        #               - (self.R3 /(self.R2+self.R3) * blind_average_h)


        #V_bol_v[r,c] = V_bol_v[r,c] if V_bol_v[r,c] > 0 else 0
        #V_bol_v[r,c] = V_bol_v[r,c] if V_bol_v[r,c] <= self.V_max else self.V_max
        #V_bol_h[r,c] = V_bol_h[r,c] if V_bol_h[r,c] > 0 else 0
        #V_bol_h[r,c] = V_bol_h[r,c] if V_bol_h[r,c] <= self.V_max else self.V_max

    print(V_bol)
    print(V_bol_h)
    print(V_bol_v)

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
    #print(prefix + "V_bol")
    #print(prefix + "V_bol_h")
    #print(prefix + "V_bol_v")
    try:
      return {"V_bol"   : np.loadtxt(prefix + "V_bol"),
              "V_bol_h" : np.loadtxt(prefix + "V_bol_h"),
              "V_bol_v" : np.loadtxt(prefix + "V_bol_v")}
    except:
      return None

  def store_display(self, prefix, args, input_data, output_data, cached):
    for key in output_data:
      fname = prefix + key + '.png'
      if not os.path.exists(fname) or not cached:
        output_normalized = 255.0*output_data[key]/(output_data[key].max())
        image_data = output_normalized.astype(np.uint8)
        #image_data = cv2.equalizeHist(image_data)
        image = Image.fromarray(image_data)
        image.save(fname)

  def get_parameter_id_str(self, args, input_data):
    T_amb = args['T_amb'] if args and args['T_amb'] else self.T_amb
    t_int = args['t_int'] if args and args['t_int'] else self.t_int
    return str(T_amb) + "_" + str(t_int)
