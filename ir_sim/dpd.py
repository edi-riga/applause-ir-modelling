import numpy as np

from . import sys_param as sp


def dpd_chan(data):
    if len(data.shape) == 3:
        frames, pix_v, pix_h = data.shape
        out = np.zeros((pix_v, pix_h))
        for f in range(frames):
            print(f'Processing frame {f+1}/{frames}')
            out += dpd_chan(data[f])
        return out >= 1
    
    v, h = data.shape
    dead = np.zeros(data.shape)
    
    for vv in range(v):
        for hh in range(h):
            if hh == 0 or vv == 0 or hh == h-1 or vv == v-1:
                continue
            
            pixels = data[vv-1:vv+2, hh-1:hh+2]
            adj_mask = np.array([[0,0,0],
                                 [0,1,0],
                                 [0,0,0]])
            adj = np.ma.masked_array(pixels, adj_mask)
            asc = np.sort(adj.reshape(adj.size))
            
            pix1 = asc[-3]  # second highest, asc[-1] is masked
            pix2 = asc[1]  # second lowest
            
            # average of remaining pixels
            avg = np.average(np.ma.masked_array(asc, [0,1,0,0,0,0,1,0,0]))
            
            val_max = avg + pix1 - pix2
            val_min = np.abs(avg - pix1 + pix2)
            #print(f'{val_max}, {val_min}')
            
            if not (val_min <= data[vv][hh] and val_max >= data[vv][hh]):
                dead[vv][hh] = 1
                #print(f'not {val_min} < {data[vv][hh]} < {val_max}')
    return dead 


def dpd_takam(data, maxval=30):
    if len(data.shape) == 3:
        frames, pix_v, pix_h = data.shape
        out = np.zeros((pix_v, pix_h))
        for f in range(frames):
            print(f'Processing frame {f+1}/{frames}')
            out += dpd_takam(data[f], lim, pix_num, d, L)
        return out >= 1
    v, h = data.shape
    dead = np.zeros(data.shape)
    
    for vv in range(v):
        for hh in range(h):
            if vv == v-1 or vv == 0 or hh == h-1 or hh == 0:
                continue
            
            est_num = data[vv-1, hh] + data[vv+1, hh] + \
                  data[vv, hh-1] + data[vv, hh+1] + \
                  (data[vv-1, hh-1] + data[vv-1, hh+1] + \
                  data[vv+1, hh-1] + data[vv+1, hh+1])/np.sqrt(4)
            est = est_num / (4 + 4/np.sqrt(2))
            dif = np.abs(est - data[vv, hh])
            avg = 1/16 * (data[vv-1, hh-1] + data[vv-1, hh] + data[vv-1, hh+1] + \
                  data[vv, hh+1] + data[vv+1, hh+1] + data[vv+1, hh] + \
                  data[vv+1, hh-1] + data[vv, hh-1] + 8*data[vv, hh])
            if dif > avg or dif > (maxval - avg):
                dead[vv, hh] = 1
    return dead

def testdpd(data, lim=sp.lim, d=5, L=7):
    if len(data.shape) == 3:
        frames, pix_v, pix_h = data.shape
        out = np.zeros((pix_v, pix_h))
        for f in range(frames):
            print(f'Processing frame {f+1}/{frames}')
            out += testdpd(data[f], lim, pix_num, d, L)
        return out >= 1
    
    v, h = data.shape
    data2d = data.reshape(data.size)
    dead = np.zeros(data.size)
    data2d = np.concatenate([data2d[:L][::-1], data2d, data2d[-L:][::-1]])
    
    i = 0
    
    while i < (h*v - L):
        PSDPD = data2d[i:i+L*2+1]  # subset
        MA = np.average(PSDPD)  # moving average
        DPV = np.abs(PSDPD - MA)
        
        if np.any(np.abs(DPV > lim*MA)):
            # Second stage
            for px in range(len(PSDPD)):
                px_addr = i + px
                adj_hor = np.array([data2d[px_addr-1], data2d[px_addr+1]])
                if (real_addr := px_addr - L) < h:
                    adj_ver = np.array([data2d[px_addr+h], data2d[px_addr+h]])
                elif real_addr > h*(v-1):
                    adj_ver = np.array([data2d[px_addr-h], data2d[px_addr-h]])
                else:
                    adj_ver = np.array([data2d[px_addr-h], data2d[px_addr+h]])
                
                est = np.average(adj_hor)
                dif = np.abs(data2d[px_addr] - est)
                avg = np.average(np.concatenate([adj_hor, adj_ver, np.array([data2d[px_addr], data2d[px_addr], data2d[px_addr], data2d[px_addr]])]))
                HDPV = DPV[DPV == DPV.max()][0]
                
                #print(f'est={est}, dif={dif}, avg={avg}, HDPV={HDPV}')
                if dif > np.abs(HDPV - avg):
                    dead[real_addr] = 1
        
        i += d
    
    return dead.reshape((v, h))


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
            
            subset = Buff[r][col : col + extend]
            ma = np.average(subset)
            tp = Buff[r][half_pix_num + col]
            dif_abs = np.abs(tp - ma)
            th = ma * lim
            if (dif_abs >= th):  # First stage check
                #print(f'{r}, {col}')
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
                #print("est =",est,"avg =",avg,"dif =",dif,"hdpv =", hdpv, "check =", check)
                if dif >= check:  # Second stage check
                    #print('dead')
                    dp_map[r][col] = 1

    return dp_map


def replace_dead(data, dp_map):
    out = np.zeros(data.shape)
    
    if (dims := len(data.shape)) == 3:
        for f in range(len(data)):
            out[f] = replace_dead(data[f], dp_map)
    elif dims == 2:
        vv, hh = data.shape
        # Ignore NaNs
        avg = np.average(data[np.isfinite(data)])
        for r in np.arange(0, vv, 1):
            for c in np.arange(0, hh, 1):
                out[r][c] = avg if dp_map[r][c] else data[r][c]
    return out


def replace_dead_conv(data, dp_map, ksize=3):
    out = np.zeros(data.shape)
    
    if (dims := len(data.shape)) == 3:
        for f in range(len(data)):
            out[f] = replace_dead_conv(data[f], dp_map, ksize)
    elif dims == 2:
        d1 = data.reshape(data.size)
        d1f = np.ma.masked_array(d1, mask=dp_map)
        avgd1 = np.convolve(d1f, np.ones(ksize), 'same') / ksize
        avg = avgd1.reshape(data.shape)
        out = np.where(dp_map, avg, data)
    return out


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

