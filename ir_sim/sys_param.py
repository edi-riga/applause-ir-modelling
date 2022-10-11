from scipy.constants import c, h, k, pi
import numpy as np

# Dimensional parameters of the sensor
pix_h = 640     # Amount of pixels in horizontal dimension
pix_v = 480     # Amount of pixels in vertical dimension


boundary_pix = 2  # Amount of rings of boundary pixels around active pixels
skimming_pix = 1  # Amount of rings of skimming pixels around active pixels

# Amount of all pixels in horizontal dimension, including boundary and skiming.
pix_h_all = pix_h + boundary_pix*2 + skimming_pix*2
# Amount of all pixels in vertical dimension, including boundary and skimming.
pix_v_all = pix_v + boundary_pix*2 + skimming_pix*2


pitch = 17e-6         # Pitch between pixels
Fill_factor = 0.65    # Fill factor

l_h = pitch * pix_h   # Horizontal size
l_v = pitch * pix_v   # Vertical size
Ar = l_h * l_v        # Sensor area m2

A_pix = pitch ** 2   # One pixel area 17x17 um2
A_sens = A_pix * Fill_factor    # Sensetive area of one pixel
Pix_area_coef = A_sens/Ar       # Coeficient for sensing area of one pixel
                                # regarding to area of whole sensor.


# Physical parameters of the sensor
R_ta_i = 1e6      # Resistance at ambient temperature (nominal)
T_sa = 300         # Ambient temperature
g_ini = 1e-7      # Thermal conductivity (nominal)
c_ini = 5e-9      # Thermal capacity (nominal)
alpha = -0.03     # TCR at ambient temperature

# Default tolerance values
R_tol = g_tol = c_tol = 1e-5

Ea = -(alpha * k * T_sa**2)                      # Activation energy calculation
R_0_dark = R_ta_i / (np.exp(Ea / (k * T_sa)))    # R0 calculation

Phi_r = 0    # Angle between the perpendicular to receiver surface
             # and line connecting receiver and radiator centers
Phi_s = 0    # Angle between the perpendicular to radiator surface
             # and line connecting receiver and radiator centers

FOV = np.pi/6      # Field of View
Thetta = FOV/2     # Half of FOV angle
fl = l_h / ( 2 * np.arctan(FOV / 2) )       # Focal length
Omega = pi * ( np.sin( Thetta ) ) ** 2   # Projected solid angle

tau = 1    # Composite transmittance of enviroment

int_time = 10e-3  # Integration time
Ib = 50e-6        # BIAS current

# Wave length of interest 8-14 um
lambd1 = 8e-6
lambd2 = 14e-6


# DPD
lim = 0.6     # Threshold for DPD stage 1
pix_num = 15  # 1D convolution kernel size.

# ROIC
# RESISTORS ON OP AMP INPUTS
R1 = 200e6
R2 = 10e3
R3 = 400e6
# GAIN CAPACITOR OF INTEGRATOR
C = 0.2e-12
# EXTERNAL VOLTAGE AS REFERENCE (NOT ACCOMPLISHED!)
V_ref = 1.65 

# ANALOG TO DIGITAL CONVERTER
max_val = 255;
max_analog = 3.2;
adc_coef = max_val/max_analog
