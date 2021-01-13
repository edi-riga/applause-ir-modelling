import sys_param as sp
import numpy as np
from numpy import exp
import matplotlib.pyplot as plt
from scipy.optimize import fsolve
from scipy.constants import k
from time import gmtime, strftime
import argparse
import os, fnmatch


def vars():
  global Ts, gi, ci, Ea, R0, itime, I, tau
  global R1, R2, R3, C
  Ts = sp.T_sa
  gi = sp.g_ini
  ci = sp.c_ini
  Ea = sp.Ea
  R0 = sp.R_0_dark
  itime = sp.int_time
  I = sp.Ib
  tau = ci/gi
  R1 = sp.R1
  R2 = sp.R2
  R3 = sp.R3
  C = sp.C


def take_file():
  global data, it
  dir = 'data_files'
  pattern = 'pto1pix*'
  list_of_files = []
  try:
    for file in os.listdir('data_files'):
      if fnmatch.fnmatch(file, pattern):
        list_of_files.append(file)
  except:
    print("Error!")
  else:
    print("Choose file to process!:")
    i=1
    for file in list_of_files:
      print(i, file)
      i += 1
    choise = int(input())
    chosen_file = dir + '/' + list_of_files[choise - 1]
    print("File %s was chosen." % chosen_file)
    data = np.loadtxt(chosen_file)
    it = np.arange(data.size)
    global V_skim_int, V_skim, V_temp, V_act, V_out_a, V_out_t, V_diff
    V_skim_int = np.ones(data.size)
    V_skim = np.ones(data.size)
    V_temp = np.ones(data.size)
    V_act = np.ones(data.size)
    V_out_a = np.ones(data.size)
    V_out_t = np.ones(data.size)
    V_diff = np.ones(data.size)
    global Pcam
    Pcam = data[0]*0.7



def pixel_values(i):
  # SKIMMING PIXELS
  V_skim_int[i] =  fsolve(lambda Va: Va - I * R0 * exp(Ea / (k * Ts + k * ((I * Va)/gi) * \
                    (1+(tau / itime) * (exp(-itime / tau) - 1)))), 0)
  V_skim[i] = fsolve(lambda Vs: Vs - I * R0 * exp(Ea / ( k * (Ts + (I*Vs/gi) * \
                    (1 - exp(-itime/tau))))), 0)
  # BOUNDARY PIXELS
  V_temp[i] = fsolve(lambda Va: Va - I * R0 * exp(Ea / (k * Ts + k * ((I * Va +\
                    Pcam) / gi) * (1 + (tau / itime) * (exp(-itime / tau) - 1)))), 0)
  # ACTIVE PIXELS
  V_act[i] = fsolve(lambda Va: Va - I * R0 * exp(Ea / (k * Ts + k * ((I * Va + data[i] + Pcam) / gi) \
                     * (1 + (tau / itime) * (exp(-itime / tau) - 1)))), 0)


def roic(i):
  V_out_a[i] = (1/(R1*C)) * ((R3/(R2+R3)) * V_skim_int[i] - V_act[i])  + (R3/(R2+R3)) * V_skim[i]
  V_out_t[i] = (1/(R1*C)) * ((R3/(R2+R3)) * V_skim_int[i] - V_temp[i]) + (R3/(R2+R3)) * V_skim[i]
  if V_out_a[i] < 0:
    V_out_a[i] = 0
  elif V_out_a[i] >= 3.2:
    V_out_a[i] = 3.2
  V_diff[i] = V_out_a[i] - V_out_t[i]


def plot_pix():
  plt.plot(it, V_out_a, '-b', it, V_out_t, '-r', it, V_diff, '-g')
  plt.grid(True)
  plt.show()


def run_pixels():
  for i in it:
    pixel_values(i)


def run_roic():
  for i in it:
    roic(i)


def run():
  run_pixels()
  run_roic()
  plot_pix()


def run_on_cam(Tsens):
  global Ts, Pcam
  Ts = Tsens
  for i in it:
    Pcam = data[i]*0.7
    run_pixels()
    run_roic()
    plt.plot(it, V_out_a, '-b', it, V_out_t, '-r', it, V_diff, '-g')
  plt.grid(True)
  plt.show()


def run_on_temp():
  global Ts, Pcam
  Pcam = data[0]*0.7
  T_list = np.arange(300, 405, 5)
  for T in T_list:
    Ts = T
    run_pixels()
    run_roic()
    plt.plot(it, V_out_a, '-b', it, V_out_t, '-r', it, V_diff, '-g')
  plt.grid(True)
  plt.show()


if __name__ == '__main__':
  vars()
  take_file()
  for i in it:
    pixel_values(i)
    roic(i)
  plot_pix()