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
from Readout    import Readout
from ADC        import ADC
from NUC        import NUC

# models
from Blackbody import Blackbody


# General parameters
FOV = np.pi/6

# TODO: check correctness
angle_projected_solid = np.pi * (np.sin(FOV/2)) ** 2

# Sensor parameters
bol_dim_h        = 17e-6
bol_dim_v        = 17e-6
bol_fill_factor  = 0.65
resolution_h     = 640
resolution_v     = 480
size_h           = bol_dim_h*resolution_h
size_v           = bol_dim_v*resolution_v
focal_length     = size_h / ( 2 * np.arctan(FOV/2) )


# Model - Black body
blackbody = Blackbody(
  area  = bol_dim_h * bol_dim_v * bol_fill_factor,
  omega = np.pi * (np.sin(FOV/2)) ** 2)

# Model - Optics
optics = Optics(
  resolution   = (resolution_h, resolution_v),
  focal_length = focal_length,
  pitch        = bol_dim_h,
  visualize    = True)

# Model - Bolometers
bolometers = Bolometers(
  size_active   = (resolution_h, resolution_v),
  visualize     = True)

# Model - Readout
readout = Readout(
  size_active   = (resolution_h, resolution_v),
  visualize     = True)

# Model - ADC 
adc = ADC(
  size_active   = (resolution_h, resolution_v),
  skim          = 'h')


# Calculate NUC coefficients
temps = [300, 400]
blackbody.set_args_list(temps)

sim_nuc_coef = Simulation([blackbody, optics, bolometers, readout, adc], use_cache=True)
output_nuc_coef = sim_nuc_coef.process()

frame0 = output_nuc_coef[0][0][0][0][0]['ADC']
frame1 = output_nuc_coef[1][0][0][0][0]['ADC']

nuc_a, nuc_b = NUC.calculate_coefs([frame0, frame1], [np.mean(frame0), np.mean(frame1)])

nuc = NUC(
  coef_a = nuc_a,
  coef_b = nuc_b,
  resolution = (resolution_h, resolution_v),
  visualize = True
)

# Run silumation with NUC
blackbody.set_args_list([t for t in range(300, 400, 10)])

sim = Simulation([blackbody, optics, bolometers, readout, adc, nuc], use_cache=True)
output = sim.process()


print(output)
