# Black body temperature in Kelvin
T = 300

# Wavelength range of interest
lambd = (8e-6, 14e-6)

# Angle of incidence
phi = (0, 0)

# Receiver's pixel sensitive area
bol_dim_h = 17e-6
bol_dim_v = 17e-6
area = bol_dim_h * bol_dim_v 

# Projected solid angle
omega = 0.75


# Resolution of the sensor's active area
resolution = (320, 240)

focal_length = 0.02

# Pitch between pixels
pitch = 17e-6


# Camera temperature
Tcam = 30

# boundary pixels
size_boundary = (2, 2, 2, 2)

# blind pixels
size_blind = (1, 1, 1, 1)

# Microbolometer parameters, median and tolerance

# Resistance at ambient temperature
R_ambient_med = 1e6
R_ambient_tol = 1e-5

# Thermal conductivity
G_thermal_med = 1e-7
G_thermal_tol = 1e-5

# Thermal capacity
C_thermal_med = 5e-9
C_thermal_tol = 1e-5

# Ambient temperature
T_ambient = 300

# Thermal coefficient of resistance
TCR = -0.03

# Bias current
I_bias = 50e-6

# Activation energy
E_act = 3.7277523e-20

# Integration time
t_int = 10e-3

# ADC maximum voltage
V_max = 3.2

# ADC components?
R1 = 200e6
R2 = 10e3
R3 = 400e6
C = 4e-13

adc_resolution = 10
nuc_fpart = 4
