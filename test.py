#!/usr/bin/env python3
import numpy as np
import sys
sys.path.append('backend')
sys.path.append('models')

# backend
from Model      import Model
from Simulation import Simulation
from Optics     import Optics
from Bolometers import Bolometers

# models
from Blackbody import Blackbody


# General parameters
T_black_body = 30
angle_radiator_receiver_r = 0
angle_radiator_receiver_s = 0
FOV = np.pi/6

# TODO: check correctness
angle_projected_solid = np.pi * (np.sin(FOV/2)) ** 2

# Sensor parameters
wavelength_lower = 8e-6
wavelength_upper = 14e-6
bol_dim_h        = 17e-6
bol_dim_v        = 17e-6
bol_fill_factor  = 0.65
resolution_h     = 320
resolution_v     = 240
size_h           = bol_dim_h*resolution_h
size_v           = bol_dim_v*resolution_v
focal_length     = size_h / ( 2 * np.arctan(FOV/2) )


# Model - Black body
blackbody = Blackbody(
  T     = T_black_body,
  lambd = (wavelength_lower, wavelength_upper),
  phi   = (angle_radiator_receiver_r, angle_radiator_receiver_s),
  area  = bol_dim_h * bol_dim_v * bol_fill_factor,
  omega = np.pi * (np.sin(FOV/2)) ** 2)

# Model - Optics
optics = Optics(
  resolution   = (resolution_h, resolution_v),
  focal_length = focal_length,
  pitch        = bol_dim_h,
  visualize    = True)

# Model - Molometers
bolometers = Bolometers(
  size_active   = (resolution_h, resolution_v),
  size_boundary = (2,2,2,2),
  size_blind    = (1,1,1,1),
  visualize     = True)

blackbody.set_args_list([30, 50])

#sim = Simulation([blackbody], use_cache=True)
#sim = Simulation([blackbody, optics], use_cache=True)
sim = Simulation([blackbody, optics, bolometers], use_cache=True)
output = sim.process()

print(output)
