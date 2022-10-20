import os
import numpy as np
from PIL import Image

import sys
sys.path.append('../backend')
from Model import Model
import params


class ADC(Model):
  """
  A simple Analog-to-digital converter model that converts input floating
  point voltage values to integers corresponding to digital output with
  the specified resolution and reference voltage. ADC can be connected to
  one of the readout outputs, thus choosing whether to use skimming rows,
  skimming columns or neither.

  Input data
  -----------
  V_bol :
      Raw microbolometer voltages.
  
  V_bol_h :
      Microbolometer voltages with horizontal skimming.
  
  V_bol_v :
      Microbolometer voltages with vertical skimming.
  
  Output data
  -----------
  V :
      ADC output signal.

  Initializer arguments
  ---------------------
  size_active :
      Resolution of the sensor's active area.
  
  size_boundary :
      Number of boundary pixels on each side.
  
  size_blind :
      Number of blind pixels on each side.
  
  skim :
      Can be set to 'v' or 'h' to perform vertical or
      horizontal skimming respectively.
  
  vref :
      ADC reference voltage, i.e. maximum input voltage.
  
  resolution :
      Digital output resolution in bits.
  """
  def __init__(self, size_active=params.resolution, size_boundary=params.size_boundary, size_blind=params.size_blind, skim=None, vref=params.V_max, resolution=params.adc_resolution):
    self.vref = vref
    self.resolution = resolution
    self.skim = skim
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


    super().__init__(
      input_tuple = {
        "V_bol"     : (self.size_total_h, self.size_total_v),
        "V_bol_h"   : (self.size_total_h, self.size_total_v),
        "V_bol_v"   : (self.size_total_h, self.size_total_v),
      },
      output_tuple = {
        "ADC"       : (self.size_active_h, self.size_active_v)
      }
    )
  
  def process(self, input_data=None, args=None):
    if self.skim == "h":
      V = input_data["V_bol_h"]
    elif self.skim == "v":
      V = input_data["V_bol_v"]
    else:
      V = input_data["V_bol"]
    
    slice_h = slice(self.size_blind_l + self.size_boundary_l, -self.size_boundary_r - self.size_blind_r)
    slice_v = slice(self.size_blind_t + self.size_boundary_t, -self.size_boundary_b - self.size_blind_b)
    V_act = V[(slice_v, slice_h)]
    adc_max = 2**self.resolution - 1
    V_sat = np.where(V_act < self.vref, V_act, self.vref)
    V_adc = np.round(V_sat / self.vref * adc_max)
    return {"ADC" : V_adc}
  
  def get_parameter_id_str(self, args, input_data):
    return str(args)
  
  def store(self, prefix, args, input_data, output_data):
    # TODO: add hashsum check
    fname = prefix + "ADC" + '.txt'
    np.savetxt(fname, output_data["ADC"])

  def load(self, prefix, args, input_data):
    # TODO: add hashsum check
    try:
      fname = prefix + str(input_data["ADC"]) + '.txt'
      return {"V":np.loadtxt(fname)}
    except:
      return None


