from scipy.constants import c, h, k, pi
import numpy as np

# Dimensional parameters of the sensor
pix_h = 640     # Amount of pixels in horizontal dimension
pix_v = 480     # Amount of pixels in vertical dimension

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
