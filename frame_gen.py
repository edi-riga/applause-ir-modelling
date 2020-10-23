#!/usr/bin/python3
import sys_param as sp
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.constants import k
from PIL import Image
from time import gmtime, strftime
import argparse
import os, fnmatch

# Initializing argparse
parser =  argparse.ArgumentParser(
          description='''
          Calculates integral by time of MBA output voltages.

          ''')

# Adding the arguments
group = parser.add_mutually_exclusive_group()
group.add_argument('-b', '--blackbody', nargs=4, type=float, help="Output @ different blackbody temperatures and constant sensor temperature. Insert start temperature, end temperature of blackbody, temperature increase step and sensor temperature")
group.add_argument('-s', '--sensor', nargs=4, type=float, help="Output @ different sensor temperatures and constant blackbody temperature. Insert start temperature, end temperature of sensor, temperature increase step and temperature of blacМарсианин123!kbody")

args = parser.parse_args()

# System parameters defined in "sys_param.py"
pix_h = sp.pix_h
pix_v = sp.pix_v
gi = sp.g_ini
ci = sp.c_ini
Ea = sp.Ea
R0d = sp.R_0_dark
itime = sp.int_time
I = sp.Ib
tau_i = ci/gi

# Collecting tolerance data 
try:
  c_th = np.loadtxt('tolerance_data/c_tolerance.txt').reshape((pix_v, pix_h))
  g_th = np.loadtxt('tolerance_data/g_tolerance.txt').reshape((pix_v, pix_h))
  tau = np.loadtxt('tolerance_data/tau_tolerance.txt').reshape((pix_v, pix_h))
  R0 = np.loadtxt('tolerance_data/R0_tolerance.txt').reshape((pix_v, pix_h))
except OSError:
  print('Directory "tolerance_data" doesn\'t exist! \nRun "tolerance.py"')
  exit()
else:
  print('Tolerance data successfully loaded! \n')

# Strings, used in names of output files and directories
fdir = 'data_files'
fdir1 = 'frames'
pattern1 = "distribution_BB_TempRange_"
pattern2 = "pto1pix_BB_TempRange_"
pattern3 = "Solutions_BB_TempRange_"
pattern4 = "Solutions_Sens_TempRange_"
# Current date. Used in names of output data
cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
date = cur_time[8:10] + '_' + cur_time[5:7]

# Creating directories
try:
  os.mkdir(fdir)
except OSError:
  print('\nDirectory "%s" already exist\n' % fdir)
else:
  print('\nSuccessfully created the directory "%s" \n' % fdir)

try:
  os.mkdir(fdir1)
except OSError:
  print('\nDirectory "%s" already exist\n' % fdir1)
else:
  print('\nSuccessfully created the directory "%s" \n' % fdir1)


#Blackbody temperature
if args.blackbody:
  
  # Blacbody temperatures
  T_start = args.blackbody[0]
  T_stop = args.blackbody[1]
  T_step = args.blackbody[2]
  
  # Temperature of sensor
  T_s = args.blackbody[3]
  
  # Creating array of blackbody temperatures
  T_list = np.arange(T_start, T_stop+T_step, T_step)
  
  # Run "blackbody.py" script in Linux Terminal
  command = 'python3 blackbody.py ' + str(T_start) + ' ' + str(T_stop) + ' ' + str(T_step) + ' -d'
  print('Calling script:')
  print(command, '\n')
  os.system(command)
  
  # Names for output data files and directories
  filename1 = fdir + '/' + pattern1 + str(T_start) + '_' + str(T_stop) + '_' + str(T_step) + '.txt'
  filename2 = fdir + '/' + pattern2 + str(T_start) + '_' + str(T_stop) + '_' + str(T_step) + '.txt'
  
  dirname1 = fdir + '/' + 'solutions_BB_Temps'
  dirname2 = fdir1 + '/' + fdir1 + '_BB_Temps_@_sensTemp_' + str(T_s) + '_' + date
  
  file_of_sol = dirname1 + '/' + pattern3 + str(T_start) + '_' + str(T_stop) + '_' + str(T_step) + '_sensTemp_' + str(T_s) + '.txt'
  
  # Creating directories
  try:
    os.mkdir(dirname1)
  except OSError:
    print('\nDirectory "%s" already exist\n' % dirname1)
  else:
    print('\nSuccessfully created the directory "%s" \n' % dirname1)
  
  try:
    os.mkdir(dirname2)
  except OSError:
    print('\nDirectory "%s" already exist\n' % dirname2)
  else:
    print('\nSuccessfully created the directory "%s" \n' % dirname2)
  
  # Loading power distribution data
  Psize = np.loadtxt(filename2).size
  P = np.loadtxt(filename1).reshape((Psize, pix_v, pix_h))
  
  # Define iteration rows for calculations
  row = np.arange(0, pix_v, 1)
  column = np.arange(0, pix_h, 1)
  
  # Define arrays for calculations and results
  V_act = np.ones((Psize, pix_v, pix_h))
  V_diff = np.ones((Psize, pix_v, pix_h))
  V_cor = np.ones((Psize, pix_v, pix_h))
  V_ampl = np.ones((Psize, pix_v, pix_h))
  V_ampl_r = np.ones((Psize, pix_v, pix_h))
  intens = np.arange(0, Psize, 1)
  
  #Coefficients for dynamic range
  x1 = 300
  x2 = 400
  y1 = 50
  y2 = 355
  n = (-1) * (y1 - y2)/(x2 - x1)
  b = (-1) * (x1*y2 - x2*y1)/(x2 - x1)
  target_values = n * T_list + b
  coef = np.zeros(T_list.shape)
  
  
  def dark_pix():
    ''' Calculates output voltage of dark pixel 
        (without IR power impinged) 
        at given bias current and sensor temperature'''
    global V_dark, I, R0d, Ea, k, T_s, gi, tau_i, itime
    V_dark = fsolve(lambda Vd: Vd-I*R0d*np.exp(Ea / (k * T_s + k*((I*Vd) / gi)*(1+(tau_i / itime)* \
    (np.exp(-itime/tau_i) - 1)  ) )), 0)
    return V_dark
  
  def active_pix(i,r,col):
    ''' Calculates output voltage of each active pixels
        at given bias current, sensor temperature and IR power impinged '''
    global V_act, I, R0d, Ea, k, T_s, gi, tau_i, itime
    V_act[i][r][col] = fsolve(lambda Va: Va - I*R0[r][col] * exp(Ea / \
    (k * T_s + k*((I*Va+P[i][r][col]) / g_th[r][col])*(1+(tau[r][col] / itime)* \
    (exp(-itime/tau[r][col]) - 1)  ) )), 0)
    return V_act
  
  def dark_active_diff(i, r, col):
    ''' Calculates difference between dark pixel
        and each active pixel voltage values '''
    global V_diff, V_dark, V_act
    V_diff[i][r][col] = (V_dark - V_act[i][r][col])
    return V_diff
  
  def correction(i):
    ''' Add ofset equal to mininal calculated V_diff value
        thus the minimal value is zero, 
        and there are no negative values in the array V_cor'''
    global V_diff, V_cor
    V_cor[i] = V_diff[i] - np.amin(V_diff[i])
    return V_cor
  
  def coefic_calc(i):
    ''' Calculate scaling coeficient for each frame '''
    global V_cor, target_values, coef
    coef[i] = target_values[i]/np.amax(V_cor[i])
    return coef
  
  def amplif_and_dyn_range(i):
    ''' Amplifying each frame to respective maximal value '''
    global V_cor, V_ampl, coef
    V_ampl[i] = V_cor[i]*coef[i]
    return V_ampl
  
  def anti_overfill(i, r, col):
    ''' Ensure that data doesn't go out of uint8 range '''
    global V_ampl, V_ampl_r
    if V_ampl[i][r][col] > 255:
      V_ampl_r[i][r][col] = 255
    elif V_ampl[i][r][col] < 0:
      V_ampl_r[i][r][col] = 0
    else:
      V_ampl_r[i][r][col] = V_ampl[i][r][col]
    return V_ampl_r
  
  # Calculations
  dark_pix()

  for i in intens:
    cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    cur_time = cur_time[11:]
    print("\nTime:", cur_time)
    print("Calculating frame @ T =", T_list[i])
    for r in row:
      for col in column:
        active_pix(i,r,col)
        dark_active_diff(i,r,col)
    correction(i)
    coefic_calc(i)
  
  for i in intens:
    amplif_and_dyn_range(i)
        
  for i in intens:
    for r in row:
      for col in column:
        anti_overfill(i, r, col)
    
    # Generating images from calculated data array
    print("Generating image @ T =", T_list[i])
    it = str(T_list[i])
    imname = dirname2 + '/' + 'nu' + it + '.png'
    Pic = V_ampl_r[i].astype(np.uint8)
    img = Image.fromarray(Pic)
    img.save(imname) 
    print('Image %s saved' % imname)
  
  # Saving data to .txt file
  with open(file_of_sol,'w') as outf:
    for data in V_ampl_r:
      np.savetxt(outf, data , fmt='%-.18e')
      
  
  
# Sensor temperature
elif args.sensor:
  # Sensor temperatures
  T_start = args.sensor[0]
  T_stop = args.sensor[1]
  T_step = args.sensor[2]
  
  # Blackbody temperature
  T_b = args.sensor[3]
  
  # Run "blackbody.py" script in Linux Terminal
  command = 'python3 blackbody.py ' + str(T_b) + ' ' + str(T_b) + ' ' + '1' + ' -d -s'
  print('Calling script:')
  print(command, '\n')
  os.system(command)
  
  # Names for output data files and directories
  filename1 = fdir + '/' + pattern1 + str(T_b) + '_' + str(T_b) + '_' + '1.0' + '.txt'
  
  dirname1 = fdir + '/' + 'solutions_sensTemps'
  dirname2 = fdir1 + '/' + fdir1 + '_' + 'sensTemps_@_BBtemp_' + str(T_b)
  
  file_of_sol = dirname1 + '/' + pattern4 + str(T_start) + '_' + str(T_stop) + '_' + str(T_step)+ '_BBtemp_'+ str(T_b) + '.txt'
  
  # Creating directories
  try:
    os.mkdir(dirname1)
  except OSError:
    print('\nDirectory "%s" already exist\n' % dirname1)
  else:
    print('\nSuccessfully created the directory "%s" \n' % dirname1)
  
  try:
    os.mkdir(dirname2)
  except OSError:
    print('\nDirectory "%s" already exist\n' % dirname2)
  else:
    print('\nSuccessfully created the directory "%s" \n' % dirname2)
  
  # Loading power distribution data
  P = np.loadtxt(filename1).reshape(pix_v, pix_h)
  
  # Creating list of sensor temperatures
  T_s = np.arange(T_start, T_stop+T_step, T_step)
  
  # Define iteration rows for calculations
  row = np.arange(0, pix_v, 1)
  column = np.arange(0, pix_h, 1)
  
  # Define arrays for calculations and results
  V_act = np.ones((T_s.size, pix_v, pix_h))
  V_diff = np.ones((T_s.size, pix_v, pix_h))
  V_cor = np.ones((T_s.size, pix_v, pix_h))
  V_ampl = np.ones((T_s.size, pix_v, pix_h))
  V_ampl_r = np.ones((T_s.size, pix_v, pix_h))
  V_dark = np.zeros(T_s.size)
  intens = np.arange(0, T_s.size, 1)
  
  #Coefficients for dynamic range
  x1 = 300
  x2 = 400
  y1 = 50
  y2 = 355
  n = (-1) * (y1 - y2)/(x2 - x1)
  b = (-1) * (x1*y2 - x2*y1)/(x2 - x1)
  target_value = n * T_b + b
  
  def dark_pix(i):
    ''' Calculates output voltage of dark pixel 
        (without IR power impinged) 
        at given bias current and sensor temperature''' 
    global V_dark, I, R0d, Ea, k, T_s, gi, tau_i, itime
    V_dark[i] = fsolve(lambda Vd: Vd-I*R0d*np.exp(Ea / (k * T_s[i] + k*((I*Vd) / gi)*(1+(tau_i / itime)* \
    (np.exp(-itime/tau_i) - 1)  ) )), 0)
    print("V_dark",str(i), V_dark[i])
    return V_dark
  
  def active_pix(i,r,col):
    ''' Calculates output voltage of each active pixels
        at given bias current, sensor temperature and IR power impinged '''
    global V_act, I, R0d, Ea, k, T_s, gi, tau_i, itime
    V_act[i][r][col] = fsolve(lambda Va: Va - I*R0[r][col] * exp(Ea / \
    (k * T_s[i] + k*((I*Va+P[r][col]) / g_th[r][col])*(1+(tau[r][col] / itime)* \
    (exp(-itime/tau[r][col]) - 1)  ) )), 0)
    return V_act

  def dark_active_diff(i, r, col):
    ''' Calculates difference between dark pixel
        and each active pixel voltage values '''
    global V_diff, V_dark, V_act
    V_diff[i][r][col] = (V_dark[i] - V_act[i][r][col])
    return V_diff
  
  def correction(i):
    ''' Add ofset equal to mininal calculated V_diff value
        thus the minimal value is zero, 
        and there are no negative values in the array V_cor'''
    global V_diff, V_cor
    V_cor[i] = V_diff[i] - np.amin(V_diff[i])
    return V_cor
  
  def coefic_calc():
    ''' Calculate scaling coeficient for each frame '''
    global V_cor, target_value, coef
    coef = target_value/np.amax(V_cor)
    return coef
    
  def amplif_and_dyn_range(i):
    ''' Amplifying each frame to respective maximal value '''
    global V_cor, V_ampl, coef
    V_ampl[i] = V_cor[i]*coef
    return V_ampl

  def anti_overfill(i, r, col):
    ''' Ensure that data doesn't go out of uint8 range '''
    global V_ampl, V_ampl_r
    if V_ampl[i][r][col] > 255:
      V_ampl_r[i][r][col] = 255
    elif V_ampl[i][r][col] < 0:
      V_ampl_r[i][r][col] = 0
    else:
      V_ampl_r[i][r][col] = V_ampl[i][r][col]
    return V_ampl_r
  
  # Calculations
  for i in intens:
    cur_time = strftime("%Y-%m-%d %H:%M:%S", gmtime())
    cur_time = cur_time[11:]
    print("\nTime:", cur_time)
    print("Calculating frame @ sensor T =", T_s[i])
    # Calculating dark pixel values at each given sensor temperature
    dark_pix(i)
    for r in row:
      for col in column:
        active_pix(i,r,col)
        dark_active_diff(i, r, col)
    correction(i)
  print('Done')
  coefic_calc()
  print('Coefficient is %f' % coef)
  for i in intens:
    amplif_and_dyn_range(i)
  
  for i in intens:
    for r in row:
      for col in column:
        anti_overfill(i, r, col)
    # Generating images from calculated data array
    print("Generating image @ T =", T_s[i])
    it = str(T_s[i])
    imname = dirname2 + '/' + 'nus'+ it + '.png'
    Pic = V_ampl_r[i].astype(np.uint8)
    img = Image.fromarray(Pic)
    img.save(imname) 
    print('Image %s saved' % imname)
  
  # Saving calculated data to .txt file
  with open(file_of_sol,'w') as outf:
    for data in V_ampl_r:
      np.savetxt(outf, data , fmt='%-.18e')

