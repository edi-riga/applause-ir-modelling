import sys
# Directory where the Non Uniformity simulator is located:
path = 'F:/PyCharm/IR/non_uniformity_sim'
''' Put this script "nuc.py" alongside with 
    Non Uniformity simulator and name the folder "non_unif_cor"'''
path_nuc = 'F:/PyCharm/IR/non_unif_cor'
sys.path.append(path)
import numpy as np
from numpy.random import randint
import os
import sys_param as sp
from PIL import Image
import matplotlib.pyplot as plt
import argparse

# Initializing argparse
parser = argparse.ArgumentParser(
  description='''
          2 and 3 point Non Uniformity Correction ''')

group = parser.add_mutually_exclusive_group()
group.add_argument('-2', '--two', action="store_true", help='Two point NUC')
group.add_argument('-3', '--three', action="store_true", help='Three point NUC')
parser.add_argument('-cs', '--calcsave', action="store_true",
                    help='Calculates coefficiens for NUC and saves in .txt file')
parser.add_argument('-n', '--nuc', action="store_true",
                    help='Calculates coefficients for NUC, makes correction of input data and saves corrected frames in .png files')
parser.add_argument('-rp', '--randomplot', nargs=1, type=int,
                    help='''Plot randomly chosen pixel values depending on BB temperature, before and after NUC.
                            Example "-rp 5" will chose 5 random pixels.''')

if len(sys.argv) == 1:
  parser.print_help(sys.stderr)
  sys.exit(1)
args = parser.parse_args()


def collect_data():
  # Collect necessary data from sys_param.py
  global bb_temp_fs, pix_h, pix_v, row, column
  path_to_data = path + '/data_files/solutions_BB_Temps'
  bb_temp_fs = os.listdir(path_to_data)
  bb_temp_fs.sort()
  pix_h = sp.pix_h
  pix_v = sp.pix_v
  row = np.arange(0, pix_v, 1)
  column = np.arange(0, pix_h, 1)


def choose_file():
  # Choose data from file, previously made by Non Uniformity Simulator:
  global choise
  print("Choose file from list:")
  nr = 1
  for f in bb_temp_fs:
    print(str(nr), '.', f)
    nr += 1
  try:
    choise = int(input("Enter the number of file: "))
  except ValueError:
    print("Please, enter the number of file, NOT name!")
    print("Please, try again.")
    choose_file()
  else:
    return choise


def take_file():
  # Get string from chosen file
  global chosen_file
  try:
    chosen_file = bb_temp_fs[choise - 1]
  except IndexError:
    print("There is no file with such number!")
    print("Please, try again.")
    choose_file()
    take_file()
  else:
    print('\n Chosen file')
    print(chosen_file)
    return chosen_file


def find_frames(data_file):
  # Collect data for NUC
  global temper_str, data, start_frame, stop_frame, temp_ar, point_temps
  path1 = path + '/data_files/'
  pattern3 = '/solutions_BB_Temps/'
  temper_str = data_file[23:39]
  T_start = float(temper_str[0:5])
  T_stop = float(temper_str[6:11])
  T_step = float(temper_str[12:16])
  temp_ar = np.arange(T_start, T_stop + T_step, T_step)
  file_of_data = path1 + pattern3 + data_file
  frame_nr = temp_ar.size
  start_frame = np.zeros((pix_v, pix_h))
  stop_frame = np.zeros((pix_v, pix_h))
  data = np.loadtxt(file_of_data).reshape((frame_nr, pix_v, pix_h))
  if args.two:
    start_frame = data[0]
    stop_frame = data[frame_nr - 1]
    point_temps = np.array([T_start, T_stop])
    print("Point tempers: \n", point_temps)
  elif args.three:
    global mid_frame
    mid_frame = np.zeros((pix_v, pix_h))
    start_frame = data[0]
    stop_frame = data[frame_nr - 1]
    if frame_nr % 2:
      mid_frame = data[frame_nr // 2]
      point_temps = np.array([T_start, temp_ar[(temp_ar.size) // 2], T_stop])
      print("Point tempers: \n ", point_temps)
    elif not frame_nr % 2:
      mid_frame = data[frame_nr // 2 - 1]
      point_temps = np.array([T_start, temp_ar[(frame_nr) // 2 - 1], T_stop])
      print("Point tempers: \n ", point_temps)


def coeff_calc():
  # Calculate the koefficients for each pixel to perform NUC
  global k_avg, b_avg
  A = np.average(start_frame)
  B = np.average(stop_frame)
  if args.two:
    global  k_nuc, b_nuc
    k_avg = (-1) * (A - B) / (point_temps[1] - point_temps[0])
    b_avg = (-1) * (point_temps[0] * B - point_temps[1] * A) / (point_temps[1] - point_temps[0])
    k_pix = (-1) * (start_frame - stop_frame) / (point_temps[1] - point_temps[0])
    b_pix = (-1) * (point_temps[0] * stop_frame - point_temps[1] * start_frame) / (point_temps[1] - point_temps[0])
    k_nuc = k_avg / k_pix
    b_nuc = (-1) * (k_avg * b_pix) / k_pix + b_avg
  elif args.three:
    global k_nuc1, b_nuc1, k_nuc2, b_nuc2
    M = np.average(mid_frame)  # Optional average value at the middle of Temps
    k_avg = (-1) * (A - B) / (point_temps[2] - point_temps[0])
    b_avg = (-1) * (point_temps[0] * B - point_temps[2] * A) / (point_temps[2] - point_temps[0])
    k_pix1 = (-1) * (start_frame - mid_frame) / (point_temps[1] - point_temps[0])
    b_pix1 = (-1) * (point_temps[0] * mid_frame - point_temps[1] * start_frame) / (point_temps[1] - point_temps[0])
    k_pix2 = (-1) * (mid_frame - stop_frame) / (point_temps[2] - point_temps[1])
    b_pix2 = (-1) * (point_temps[1] * stop_frame - point_temps[2] * mid_frame) / (point_temps[2] - point_temps[1])
    k_nuc1 = k_avg / k_pix1
    b_nuc1 = (-1) * (k_avg * b_pix1) / k_pix1 + b_avg
    k_nuc2 = k_avg / k_pix2
    b_nuc2 = (-1) * (k_avg * b_pix2) / k_pix2 + b_avg


def nuc():
  # Perform 2 or 3 point NUC
  global out_unif
  out_unif = np.zeros((data.shape))
  it = np.arange(len(data))
  if args.two:
    for i in it:
      out_unif[i] = k_nuc * data[i] + b_nuc
  elif args.three:
    row = np.arange(pix_v)
    column = np.arange(pix_h)
    for i in it:
      for r in row:
        for c in column:
          if (data[i][r][c] <= mid_frame[r][c]):
            out_unif[i][r][c] = k_nuc1[r][c] * data[i][r][c] + b_nuc1[r][c]
          elif (data[i][r][c] > mid_frame[r][c]):
            out_unif[i][r][c] = k_nuc2[r][c] * data[i][r][c] + b_nuc2[r][c]


def make_images():
  # Create and save corrected frames
  if args.two:
    dirname = path_nuc + '/frames_after_nuc2_' + temper_str
    pattern = 'frames_after_nuc2_'
  elif args.three:
    dirname = path_nuc + '/frames_after_nuc3_' + temper_str
    pattern = 'frames_after_nuc3_'
  try:
    os.mkdir(dirname)
  except OSError:
    print('\nDirectory "%s" already exist\n' % dirname)
  else:
    print('\nSuccessfully created the directory "%s" \n' % dirname)
  it = np.arange(len(out_unif))
  for i in it:
    imname = pattern + temper_str + '/' + str(temp_ar[i]) + '.png'
    Pic = out_unif[i].astype(np.uint8)
    Pic = Image.fromarray(Pic)
    Pic.save(imname)


def save_data():
  # Save calculated coefficients to .txt file, for 2 or 3 point NUC
  row = np.arange(pix_v)
  column = np.arange(pix_h)
  if args.two:
    with open('nuc2_coeffs_out.txt', 'w') as outfile:
      ka = '%.4e' % k_avg
      ba = '%.4e' % b_avg
      format1 = '-- Coefficient for average values:\n-- k_avg      b_avg\n'
      line1 = ka + '    ' + ba + '\n'
      format2 = '-- Data format:\n-- k_nuc      b_nuc \n'
      outfile.write(format1)
      outfile.write(line1)
      outfile.write(format2)
      for r in row:
        for c in column:
          k = '%.4e' % k_nuc[r][c]
          b = '%.4e' % b_nuc[r][c]
          line2 = k + "    " + b + '\n'
          outfile.write(line2)
  elif args.three:
    with open('nuc3_coeffs_out.txt', 'w') as outfile:
      ka = '%.4e' % k_avg
      ba = '%.4e' % b_avg
      format1 = '-- Coefficient for average values:\n-- k_avg      b_avg\n'
      line1 = ka + '    ' + ba + '\n'
      format2 = '-- Data format:\n-- k_nuc1     b_nuc1         pix_thrsd     k_nuc2        b_nuc2\n'
      outfile.write(format1)
      outfile.write(line1)
      outfile.write(format2)
      for r in row:
        for c in column:
          k_n1 = '%.4e' % k_nuc1[r][c]
          b_n1 = '%.4e' % b_nuc1[r][c]
          pix = '%.4e' % mid_frame[r][c]
          k_n2 = '%.4e' % k_nuc2[r][c]
          b_n2 = '%.4e' % b_nuc2[r][c]
          line2 = k_n1 + '    ' + b_n1 + '    ' + pix + '    ' + k_n2 + '    ' + b_n2 + '\n'
          outfile.write(line2)


def plot_rnd_pix(count):
  # Plot randomly chosen pixel values
  global chosen_pixels, unif_pixels
  it = np.arange(0, count, 1)
  temp = np.arange(0, temp_ar.size, 1)
  rows = randint(0, 480, count)
  columns = randint(0, 640, count)
  chosen_pixels = np.zeros((count, temp_ar.size))
  unif_pixels = np.zeros((count, temp_ar.size))
  for i in it:
    for t in temp:
      chosen_pixels[i][t] = data[t][rows[i]][columns[i]]
      unif_pixels[i][t] = out_unif[t][rows[i]][columns[i]]
  fig, (ax1, ax2, ax3) = plt.subplots(3)
  for i in it:
    ax1.plot(temp_ar, chosen_pixels[i])
    ax2.plot(temp_ar, unif_pixels[i])
    ax3.plot(chosen_pixels[i], unif_pixels[i])
  ax1.grid(True)
  ax2.grid(True)
  ax3.grid(True)
  plt.show()

# Handle script functions
if (args.two or args.three):
  collect_data()
  choose_file()
  take_file()
  find_frames(chosen_file)
  coeff_calc()
  if args.calcsave:
    save_data()
  if args.nuc:
    nuc()
    make_images()
  if args.randomplot:
    nuc()
    plot_rnd_pix(args.randomplot[0])
else:
  pass
