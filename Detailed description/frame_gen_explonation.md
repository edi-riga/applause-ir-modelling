#### Importing all necessary and initializing argparse

```python
#!/usr/bin/python3
import sys
import sys_param as sp
import numpy as np
from numpy import exp
from numpy.random import randint
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
          The script creates data similar to not calibrated IR sensor RAW data.
          ''')


group1 = parser.add_mutually_exclusive_group()
group1.add_argument('-sr', '--skimmingrow', action='store_true',
                    help="Use skimming row as reference for integrator")
group1.add_argument('-sc', '--skimmingcolumn', action='store_true',
                    help="Use skimming column as reference for integrator")
group1.add_argument('-ev', '--externalvoltage', action='store_true',
                    help="Use external voltage as reference for integrator")
parser.add_argument('-rt', '--removetemperature', action='store_true', 
                    help='Use boundary pixels to remove temperature offset', default=0)
group2 = parser.add_mutually_exclusive_group()
group2.add_argument('-ff', '--fromfile', action='store_true',
                    help='BB IR power distribution token from file')
group2.add_argument('-rbb', '--runblackbody', nargs=3, type=str,
                    help='Calls "blackbody.py" script directly \
                    from here and create data for IR power distribution.\
                    Use three positional temperature arguments in addition: \
                    T_start T_stop T_step. Example: -rbb 300 400 20')

if len(sys.argv) == 1:
  parser.print_help(sys.stderr)
  sys.exit(1)
args = parser.parse_args()
```
----

#### Collecting data from "sys_param(.py)" module and physical parameters of pixels calculated by "tolerance(.py)"

```python
def get_system_params():
  # System parameters defined in "sys_param.py"
  global pix_h, pix_v, pix_h_all, pix_v_all, b_pix, s_pix
  pix_h = sp.pix_h
  pix_v = sp.pix_v
  pix_h_all = sp.pix_h_all
  pix_v_all = sp.pix_v_all
  b_pix = sp.boundary_pix
  s_pix = sp.skimming_pix
  global T_s, gi, ci, Ea, R0d, itime, I, tau_i
  T_s = sp.T_sa
  gi = sp.g_ini
  ci = sp.c_ini
  Ea = sp.Ea
  R0d = sp.R_0_dark
  itime = sp.int_time
  I = sp.Ib
  tau_i = ci/gi
  # ROIC COMPONENTS
  global R1, R2, R3, C, V_ref, adc_coef
  # RESISTORS ON OP AMP INPUTS
  R1 = sp.R1
  R2 = sp.R2
  R3 = sp.R3
  # GAIN CAPACITOR OF INTEGRATOR
  C = sp.C
  # EXTERNAL VOLTAGE AS REFERENCE
  V_ref = sp.V_ref
  # ADC Coefficient
  adc_coef = sp.adc_coef

  # Collecting tolerance data
  global c_th, g_th, tau, R0
  try:
    c_th = np.loadtxt('tolerance_data/c_tolerance.txt').reshape((pix_v_all, pix_h_all))
    g_th = np.loadtxt('tolerance_data/g_tolerance.txt').reshape((pix_v_all, pix_h_all))
    tau = np.loadtxt('tolerance_data/tau_tolerance.txt').reshape((pix_v_all, pix_h_all))
    R0 = np.loadtxt('tolerance_data/R0_tolerance.txt').reshape((pix_v_all, pix_h_all))
  except OSError:
    print("Directory 'tolerance_data' doesn't exist! \nRun 'tolerance.py'")
    exit()
  else:
    print('Tolerance data successfully loaded! \n')
```
----

#### Two functions below handles input data choose options
 - In case of argument '--runblackbody' and 3 additional values of 
'T_start', 'T_stop', 'T_step' srcipt 'blackbody(.py)' will be called from termilal. 
Data, produced by 'blackbody(.py)', is stored in temporary text file, which is used as
input for creating frames.

- In case of argument '--fromfile', it will be possible to choose some previously created data file as the input data for 'frame_gen(.py')

```python
def choose_file(input_list):
  global chosen
  i = 1
  for f in input_list:
    print(i, f)
    i += 1
  print("Choose file to process from list above")
  index = int(input())
  chosen = input_list[index- 1]
  print("File to process is:", chosen)


def get_data_to_process():
  global P, Psize, Pcam
  if args.runblackbody:
    global P, Psize, Pcam
    T_start = args.runblackbody[0]
    T_stop = args.runblackbody[1]
    T_step = args.runblackbody[2]
    command = 'python blackbody.py ' + T_start + ' '\
              + T_stop + ' ' + T_step + ' ' + '-d ' + '-r'
    os.system(command)
    Pcam = np.loadtxt('buf_of_powers.txt')
    Psize = Pcam.size
    P = np.loadtxt('buf_of_data.txt').reshape((Psize, pix_v, pix_h))
  elif args.fromfile:
    global list_of_files
    list_of_files = []
    filedir = 'data_files/'
    pattern1 = 'distribution_BB_TempRange*'
    pattern2 = 'pto1pix_BB_TempRange_'
    for f in os.listdir(filedir):
      if fnmatch.fnmatch(f, pattern1):
        list_of_files.append(f)
    list_of_files.sort()
    try:
      choose_file(list_of_files)
    except IndexError:
      print("Wrong number!")
      get_data_to_process(0)
    except ValueError:
      print("Wrong input!")
      get_data_to_process(0)
    else:
      path_to_file = 'data_files/' + chosen
      first_dim_file = filedir + pattern2 + chosen[len(pattern1):]
      Pcam = np.loadtxt(first_dim_file)
      Psize = Pcam.size
      P = np.loadtxt(path_to_file).reshape((Psize, pix_v, pix_h))
```
----

#### Defining iterative rows and necessary shape arrays for calculations
```python
def arrays_and_rows():
  # Define iteration rows for calculations
  global row, column, V_all, V_skim, V_out, V_out_im, V_diff, V_cor, V_cor_im, intens
  row = np.arange(pix_v_all)
  column = np.arange(pix_h_all)
  intens = np.arange(Psize)
  global row_for_out, col_for_out
  row_for_out = np.arange(pix_v_all-s_pix*2)
  col_for_out = np.arange(pix_h_all-s_pix*2)
  global row_im, col_im
  row_im = np.arange(pix_v)
  col_im = np.arange(pix_h)
  global b_pix_row
  b_pix_row = np.arange(s_pix, b_pix+s_pix, 1)

  # Define arrays for calculations and results
  V_all   = np.ones((Psize, pix_v_all, pix_h_all))
  V_skim  = np.ones((Psize, pix_v_all, pix_h_all))
  V_out = np.ones((Psize, pix_v_all-s_pix*2, pix_h_all-s_pix*2))
  V_out_im = np.ones((Psize, pix_v, pix_h))
  V_diff = np.ones((Psize, pix_v, pix_h))
  V_cor = np.zeros(V_out.shape)
  V_cor_im = np.zeros((Psize, pix_v, pix_h))
```
----

#### Date and time strings are used in naming of output dirs and files
```python
def date_and_time(pr):
  global cur_time, cur_time1, date
  cur_time = strftime("%H_%M", gmtime())
  cur_time1 = strftime("%H:%M:%S", gmtime())
  date = strftime("%m_%d", gmtime())
  if pr:
    print(cur_time1)
```
----

#### Calculating output voltages of mikrobolometers or input voltages of ROIC cells at given BIAS current time and value, and IR power radiated by blackbody.

![](Img/integrator.png)
```python
def pixel_values(i,r,col):
  ''' Calculates output voltage of each active pixels
      at given bias current, sensor temperature and IR power impinged '''
  if ((r==0) or (r<=(s_pix-1)) or (r==(pix_v_all-1)) or (r >= (pix_v_all - s_pix)) or (col == 0) or \
        (col <= (s_pix-1)) or (col >= (pix_h_all - s_pix)) or  (col == (pix_h_all -1 ))):
    # SKIMMING PIXELS
    V_all[i][r][col] = fsolve(lambda Va: Va - I * R0[r][col] * exp(Ea / (k * T_s + k * ((I * Va) / \
                              g_th[r][col]) * (1 + (tau[r][col] / itime) * (exp(-itime / tau[r][col]) - 1)))), 0)
    V_skim [i][r][col] = fsolve(lambda Vs: Vs - I * R0[r][col] * exp(Ea / ( k * (T_s + (I*Vs/g_th[r][col]) * \
                                           (1 - exp(-itime/tau[r][col]))))), 0)
  elif    ((r >= s_pix) and (r < (s_pix + b_pix))) or ((r >= (pix_v_all - s_pix - b_pix)) and \
          (r < (pix_v_all - s_pix))) or ( (r >= (s_pix + b_pix)) and (r < (pix_v_all - s_pix - b_pix)) and \
          (col >= s_pix) and (col < (s_pix + b_pix)) ) or ( (r >= (s_pix + b_pix)) and \
          (r < (pix_v_all - s_pix - b_pix)) and (col >= (pix_h_all - s_pix - b_pix)) and \
          (col < (pix_h_all - s_pix))):
    # BOUNDARY PIXELS
    V_all[i][r][col] =  fsolve(lambda Va: Va - I * R0[r][col] * exp(Ea / (k * T_s + k * ((I * Va + Pcam[0]) / \
                              g_th[r][col]) * (1 + (tau[r][col] / itime) * (exp(-itime / tau[r][col]) - 1)))), 0)
  elif (((r >= (s_pix + b_pix)) and (r < (pix_v_all - s_pix - b_pix))) and \
        ((col >= (s_pix + b_pix)) and (col < (pix_h_all - s_pix - b_pix)))):
    # ACTIVE PIXELS
    V_all[i][r][col] =  fsolve(lambda Va: Va - I*R0[r][col] * exp(Ea / (k * T_s + k*((I*Va + \
                               P[i][r - s_pix - b_pix][col - s_pix - b_pix] + Pcam[0]) / \
                               g_th[r][col])*(1+(tau[r][col] / itime) * (exp(-itime/tau[r][col]) - 1)  ) )), 0)
```
----

#### Calculating actual pixel data values (ROIC output).
Two options of reference voltage for non-inverting input of ROIC cell op amp are available now:

 - Using skimming column
 - Using skimming row

TO DO: External voltage as reference.

```python
def roic_function(i, r, col):
  if args.skimmingcolumn:
    # Skimming column is chosen
    # TO DO: PARASITIC DECREASE OF NON-INVERTING INPUT IMPEDANCE,
    # BECAUSE SKIMMING PIXELS ARE SIMULTANEOUSLY CONNECTED TO ALL INTEGRATORS IN THE ROW.
    if s_pix == 1:
      # Average value of skimming in the current row
      skimming_average_int = np.average( [V_all[i][r+s_pix][0], V_all[i][r+s_pix][pix_h_all - 1]] )
      skimming_average = np.average([V_skim[i][r+s_pix][0], V_skim[i][r+s_pix][pix_h_all - 1]])
      V_out[i][r][col] = (1/(R1*C)) * ( (R3 /(R2+R3)) * skimming_average_int \
                         - V_all[i][r+s_pix][col+s_pix] ) + (R3 /(R2+R3) * skimming_average)
    else:
      pass # TO DO: Here should be defined the ROIC function if sensor has more than 1 ring of skimming pixels

    # Integrators' analog output saturation:
    if V_out[i][r][col] < 0:
      V_out[i][r][col] = 0
    elif V_out[i][r][col] >= 3.2:
      V_out[i][r][col] = 3.2
  elif args.skimmingrow:
    # Skimming row is chosen
    # TO DO: Skimming pixels become HOT!!!
    if s_pix == 1:
      # Average value of skimming in the current row
      skimming_average_int = np.average([V_all[i][0][col+s_pix], V_all[i][pix_v_all - 1][col+s_pix]])
      skimming_average = np.average([V_skim[i][0][col+s_pix], V_skim[i][pix_v_all - 1][col+s_pix]])
      V_out[i][r][col] = (1 / (R1 * C)) * ((R3 /(R2+R3)) * skimming_average_int\
                         - V_all[i][r+s_pix][col+s_pix])+(R3 /(R2+R3)*skimming_average)
    else:
      pass # TO DO: Here should be defined the ROIC function if sensor has more than 1 ring of skimming pixels

    if V_out[i][r][col] < 0:
      V_out[i][r][col] = 0
    elif V_out[i][r][col] >= 3.2:
      V_out[i][r][col] = 3.2
  # TO DO: If external voltage is used as the reference of integrator (NON-INVERTING INPUT)
  # elif args.externalvoltage:
  #   # External voltage as reference
  #   V_out[i][r+s_pix][col+s_pix] = (1 / (R1 * C)) * (V_ext*itime - V_all[i][r+s_pix][col+s_pix])
```
----

#### Prototype function of microbolometer array thermal stabilization

Correction of output pixel values using boundary pixels. 

```python
def reject_temper_offset(i, r, col):
  bnd_aver = np.average([ V_out[i][r+s_pix + b_pix - 1][b_pix_row],V_out[i][r + s_pix + b_pix - 1][column.size - s_pix - b_pix_row - 1]])
  V_out_im[i][r][col] = V_out[i][r + s_pix + b_pix - 1][col + s_pix + b_pix - 1] - bnd_aver
```
----

#### Collecting values of active pixels to separate array, to create images of frames

```python
def actives(i, r, col):
  V_out_im[i][r][col] = V_out[i][r + s_pix + b_pix - 1][col + s_pix + b_pix - 1]
```
----

#### This way of function definition is useful during interactive python session
```python
def run_reject():
  for i in intens:
    for r in row_im:
      for col in col_im:
        reject_temper_offset(i,r,col)


def run_actives():
  for i in intens:
    for r in row_im:
      for col in col_im:
        actives(i, r, col)
```
----

#### Simpliest (linear) ADC function
Output `V_out_adc` maximum value corresponds to 
8b resolution or 0 - 254 value range in decimal.
```python
def adc():
  global V_out_adc
  V_out_adc = V_out_im * adc_coef
```
----

#### Sensor calibration operations (DPD, NUC)
Originally ADC has 14b resolution, so DPD and NUC should be performed using output decimal values in range 0 -16383, that corresponds to 14 bits min and max respectivly

To be able to create output frames using python `PIL.Image` package, it is necessary to 
convert data to 8b resolution.

TO DO: 
- Make previos function `adc()` output corresponding 14b ADC resolution
- Make function to save data in value range that corresponds to 14b resolution
- Make function to run `dpd.py` and `nuc.py`, to perform calibration with data corresponding to 14b resolution.

```python
''' Non_uniformity Calibration and Correction goes here goes here HERE!!!'''

def binary_to_image():
  pass
```
----

#### Plot chosen amount of pixels randomly choosen from array.
Function is useful in interactive session to estimate plots of pixel values as functions on IR power impinged on its area.
```python
def rand_pix_plot(amount):
  global rand_pix
  r_row = randint(0, pix_v - 1, amount)
  r_col = randint(0, pix_h - 1, amount)
  rand_pix = np.zeros((r_row.size, intens.size))
  adr = np.arange(r_row.size)
  for a in adr:
    for i in intens:
      rand_pix[a][i] = V_out_im[i][r_row[a]][r_col[a]]
    plt.plot(intens, rand_pix[a])
  plt.grid(True)
  plt.show()
```
----

#### Saving output data 
```python
def save_data():
  # Saving data to .txt file
  global dirname1
  date_and_time(0)
  dirname1 = 'data_files/dataSet_' + cur_time + '_' + date
  file_of_sol = dirname1 + '/' + 'data.txt'
  try:
    os.mkdir(dirname1)
  except OSError:
    print("Directory %s already exist." % dirname1)
  else:
    print("Directory %s created." % dirname1)

  with open(file_of_sol, 'w') as outf:
    for data in P:
      np.savetxt(outf, data)
```
----

#### Funcions, useful in interactive session
```python
def run_pixels():
  for i in intens:
    nr = i+1
    print("Processing frame Nr %s" %nr)
    date_and_time(1)
    for r in row:
      for col in column:
        pixel_values(i, r, col)


def run_roic():
  for i in intens:
    for r in row_for_out:
      for col in col_for_out:
        roic_function(i, r, col)


def run_convertion():
  global Im
  for i in intens:
    for r in row_im:
      for col in col_im:
        V_cor_im[i][r][col] = V_out_adc[i][r][col]
  Im = V_cor_im.astype(np.uint8)
```
----
#### Creating and saving images of output frames.
```python
def run_make_image(Img):
  # Generating images from calculated data array
  global dirname2
  dirname2 = dirname1 + '/' + 'frames'
  try:
    os.mkdir(dirname2)
  except OSError:
    print("Directory %s already exist." % dirname2)
  else:
    print("Directory %s is created" % dirname2)
  for i in intens:
    print("Generating image Nr. ", i + 1)
    it = str(i)
    imname = dirname2 + '/' + 'nu' + it + '.png'
    img = Image.fromarray(Img[i])
    img.save(imname)
    print('Image %s saved' % imname)
```
----
#### Useful in interactive session
```python
def run():
  run_roic()
  run_anti_overfill()
  run_convertion()
  run_make_image(Im)
```
----

#### Main():
```python
if __name__ == '__main__':
  get_system_params()
  get_data_to_process()
  arrays_and_rows()
  run_pixels()
  run_roic()
  if args.removetemperature:
    run_reject()
  else:
    run_actives()
  adc()
  run_convertion()
  save_data()
  run_make_image(Im)
```