import sys
path_sim = 'F:/PyCharm/IR/non_uniformity_sim'
sys.path.append(path_sim)
import os
import numpy as np
import sys_param as sp



def collect_data():
  global files_to_choose, pix_h, pix_v, row, column, limit, pixel_number
  path_to_data = path_sim + '/data_files/solutions_BB_Temps'
  files_to_choose = os.listdir(path_to_data)
  files_to_choose.sort()
  pix_h = sp.pix_h
  pix_v = sp.pix_v
  row = np.arange(0, pix_v, 1)
  column = np.arange(0, pix_h, 1)
  limit = sp.lim
  pixel_number = sp.pix_num

def choose_file():
  # Choose data from file, previously made by Non Uniformity Simulator:
  global choise
  print("Choose file from list:")
  nr = 1
  for f in files_to_choose:
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
    chosen_file = files_to_choose[choise - 1]
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
  global temper_str, data, temp_ar, frame_nr
  path1 = path_sim + '/data_files/'
  pattern3 = '/solutions_BB_Temps/'
  temper_str = data_file[23:39]
  T_start = float(temper_str[0:5])
  T_stop = float(temper_str[6:11])
  T_step = float(temper_str[12:15])
  temp_ar = np.arange(T_start, T_stop + T_step, T_step)
  file_of_data = path1 + pattern3 + data_file
  frame_nr = temp_ar.size
  data = np.loadtxt(file_of_data).reshape((frame_nr, pix_v, pix_h))
  #if frame_nr % 2 :


def frame_to_process(pix_num):
  global frames, Buff, half_pix_num, extend
  print("Collecting data.")
  frames = np.array([data[0], data[frame_nr//2], data[len(data)-1]])
  extend = pix_num - 1
  half_pix_num = extend/2
  half_pix_num = int(half_pix_num)
  Buff = np.zeros((len(frames), pix_v, pix_h + extend))
  nr = np.arange(len(frames))
  it = np.arange(half_pix_num)
  for n in nr:
    for r in row:
      Buff[n][r][half_pix_num: pix_h + half_pix_num] = frames[n][r]
      for i in it:
        Buff[n][r][i] = frames[n][r][half_pix_num + 1 + i]
        Buff[n][r][pix_h + half_pix_num + i] = frames[n][r][pix_h - extend - 1 + i]
  print("Done.")

def dpd(lim):
  global dp_map_one, dp_map_two
  print("Detecting dead pixels.")
  dp_map_one = np.zeros((frames.shape))
  dp_map_two = np.zeros((frames.shape))
  nr = np.arange(len(frames))

  for n in nr:
    print("Procesing frame number %s" % n)
    for r in row:
      for col in column:
        ma = np.average(Buff[n][r][col : col + extend])
        tp = Buff[n][r][half_pix_num + col]
        dif_abs = np.abs(tp - ma)
        th = ma * lim
        #print("ma =", ma, "tp =", tp, "dif_abs =", dif_abs, "th =", th)
        if (dif_abs >= th):
          #print("ma =", ma, "tp =", tp, "dif_abs =", dif_abs,"th =", th)
          dp_map_one[n][r][col] = 1
          adj_left = Buff[n][r][col - 1]
          adj_right = Buff[n][r][col + 1]
          adj_hor = np.array([adj_left, adj_right])
          if r == 1:
            adj_down = Buff[n][r+1][col]
            adj_vert = np.array([adj_down, adj_down])
          elif (r == (pix_v -1)):
            adj_up = Buff[n][r-1][col]
            adj_vert = np.array([adj_up, adj_up])
          else:
            adj_down = Buff[n][r + 1][col]
            adj_up = Buff[n][r - 1][col]
            adj_vert = np.array([adj_up, adj_down])
          est = np.average(adj_hor)
          avg = np.average(np.array([adj_hor, adj_vert]))
          dif = np.abs(tp - est)
          hdpv = np.abs(tp - ma)
          check = np.abs(avg - hdpv)
          #print("est =",est,"avg =",avg,"dif =",dif,"hdpv =", hdpv, "check =", check)
          if dif >= check:
            dp_map_two[n][r][col] = 1
  print("Done.")

def dpd_final():
  global stuck_l_dp_map, stuck_h_dp_map, dp_map_final
  print("Finalizing dead pixel map.")
  stuck_l_dp_map = np.zeros((pix_v, pix_h))
  stuck_h_dp_map = np.zeros((pix_v, pix_h))
  dp_map_final = np.zeros((pix_v, pix_h))
  for r in row:
    for col in column:
      if ((dp_map_two[2][r][col] == 1)):
        stuck_l_dp_map[r][col] = 1
        dp_map_final[r][col] = 1
      if ((dp_map_one[0][r][col] == 1)):
        stuck_h_dp_map[r][col] = 1
        dp_map_final[r][col] = 1
  print("Done.")

def save_data():
  print("Saving data")
  np.savetxt('dp_map.txt', dp_map_final, fmt='%.0d')
  print("Done")

if __name__ == '__main__':
  collect_data()
  choose_file()
  take_file()
  find_frames(chosen_file)
  frame_to_process(pixel_number)
  dpd(limit)
  dpd_final()
  save_data()