import numpy as np

from . import sys_param as sp


def dpd_(data, lim=sp.lim, pix_num=sp.pix_num):
  frame_nr, pix_v, pix_h = data.shape
  print("Collecting data.")
  frames = np.array([data[0], data[frame_nr//2], data[len(data)-1]])
  extend = pix_num - 1
  half_pix_num = extend/2
  half_pix_num = int(half_pix_num)
  Buff = np.zeros((len(frames), pix_v, pix_h + extend))
  nr = np.arange(len(frames))
  it = np.arange(half_pix_num)
  row = np.arange(0, pix_v, 1)
  column = np.arange(0, pix_h, 1)
  for n in nr:
    for r in row:
      Buff[n][r][half_pix_num: pix_h + half_pix_num] = frames[n][r]
      for i in it:
        Buff[n][r][i] = frames[n][r][half_pix_num + 1 + i]
        Buff[n][r][pix_h + half_pix_num + i] = frames[n][r][pix_h - extend - 1 + i]
  print("Done.")
  
  print("Detecting dead pixels.")
  dp_map_one = np.zeros((frames.shape))
  dp_map_two = np.zeros((frames.shape))
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
  

  print("Finalizing dead pixel map.")
  #stuck_l_dp_map = np.zeros((pix_v, pix_h))
  #stuck_h_dp_map = np.zeros((pix_v, pix_h))
  dp_map_final = np.zeros((pix_v, pix_h))
  for r in row:
    for col in column:
      if ((dp_map_two[2][r][col] == 1)):
        #stuck_l_dp_map[r][col] = 1
        dp_map_final[r][col] = 1
      if ((dp_map_one[0][r][col] == 1)):
        #stuck_h_dp_map[r][col] = 1
        dp_map_final[r][col] = 1
      # 
      if np.isnan(data[0][r][col]):
        dp_map_final[r][col] = 1
  print("Done.")
  return dp_map_final


def dpd(data, lim=sp.lim, pix_num=sp.pix_num):
    if len(data.shape) == 3:
        frames, pix_v, pix_h = data.shape
        out = np.zeros((pix_v, pix_h))
        for f in range(frames):
            print(f'Processing frame {f+1}/{frames}')
            out += dpd(data[f], lim, pix_num)
        return out >= 1
    elif len(data.shape) == 2:
        pix_v, pix_h = data.shape
    
    extend = pix_num - 1
    half_pix_num = extend // 2
    Buff = np.zeros((pix_v, pix_h + extend))
    it = np.arange(half_pix_num)
    row = np.arange(0, pix_v, 1)
    column = np.arange(0, pix_h, 1)
    for r in row:
        Buff[r][half_pix_num: pix_h + half_pix_num] = data[r]
        for i in it:
            Buff[r][i] = data[r][half_pix_num + 1 + i]
            Buff[r][pix_h + half_pix_num + i] = data[r][pix_h - extend - 1 + i]
  
    dp_map = np.zeros((pix_v, pix_h))
    
    for r in row:
        for col in column:
            if np.isnan(data[r][col]):
                dp_map[r][col] = 1
                continue
            
            ma = np.average(Buff[r][col : col + extend])
            tp = Buff[r][half_pix_num + col]
            dif_abs = np.abs(tp - ma)
            th = ma * lim
            if (dif_abs >= th):  # First stage check
                #print("ma =", ma, "tp =", tp, "dif_abs =", dif_abs,"th =", th)
                adj_left = Buff[r][col - 1]
                adj_right = Buff[r][col + 1]
                adj_hor = np.array([adj_left, adj_right])
                if r == 1:
                    adj_down = Buff[r+1][col]
                    adj_vert = np.array([adj_down, adj_down])
                elif (r == (pix_v -1)):
                    adj_up = Buff[r-1][col]
                    adj_vert = np.array([adj_up, adj_up])
                else:
                    adj_down = Buff[r + 1][col]
                    adj_up = Buff[r - 1][col]
                    adj_vert = np.array([adj_up, adj_down])
                est = np.average(adj_hor)
                avg = np.average(np.array([adj_hor, adj_vert]))
                dif = np.abs(tp - est)
                hdpv = np.abs(tp - ma)
                check = np.abs(avg - hdpv)
                if dif >= check:  # Second stage check
                    #print("est =",est,"avg =",avg,"dif =",dif,"hdpv =", hdpv, "check =", check)
                    dp_map[r][col] = 1

    return dp_map


def replace_dead(data, dp_map):
    data_out = np.zeros(data.shape)
    ff, vv, hh = data.shape
    for frame in np.arange(0, ff, 1):
        # Ignore NaNs
        avg = np.average(data[frame][np.isfinite(data[frame])])
        for r in np.arange(0, vv, 1):
            for c in np.arange(0, hh, 1):
                data_out[frame][r][c] = avg if dp_map[r][c] else data[frame][r][c]
    return data_out


def stuck_pixels(data):
    assert len(data.shape) == 3
    return data[0] == data[-1]


def saturated_pixels(data, max=sp.max_analog):
    if len(data.shape) == 3:
        frames, pix_v, pix_h = data.shape
        out = np.zeros((pix_v, pix_h))
        for f in range(frames):
            print(f'Processing frame {f}/{frames}')
            out += saturated_pixels(data[f], max)
        return out >= 1
    elif len(data.shape) == 2:
        pix_v, pix_h = data.shape
    
    out = np.zeros((pix_v, pix_h))
    for r in range(pix_v):
        for col in range(pix_h):
            out[r][col] = data[r][col] == max
    return out

