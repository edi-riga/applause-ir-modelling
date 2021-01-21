import sys
import numpy as np

from . import sys_param as sp

def coeff_calc_quad(frames):
    start_frame = frames[0]
    stop_frame = frames[-1]
    A = np.average(start_frame)
    B = np.average(stop_frame)
    n = len(frames) // 2 if len(frames) % 2 else len(frames) // 2 - 1
    mid_frame = frames[n]
    M = np.average(mid_frame)
    
    a_nuc = np.zeros(frames[0].shape)
    b_nuc = np.zeros(frames[0].shape)
    c_nuc = np.zeros(frames[0].shape)
    _, v, h = frames.shape
    
    for vv in range(v):
        for hh in range(h):
            a_nuc[vv, hh] = (-start_frame[vv, hh]*M + start_frame[vv, hh]*B + mid_frame[vv, hh]*A \
                - mid_frame[vv, hh]*B - stop_frame[vv, hh]*A + stop_frame[vv, hh]*M) \
                / (start_frame[vv, hh]**2*mid_frame[vv, hh] - start_frame[vv, hh]**2*stop_frame[vv, hh] \
                - start_frame[vv, hh]*mid_frame[vv, hh]**2 + start_frame[vv, hh]*stop_frame[vv, hh]**2 \
                + mid_frame[vv, hh]**2*stop_frame[vv, hh] - mid_frame[vv, hh]*stop_frame[vv, hh]**2)
            b_nuc[vv, hh] = (start_frame[vv, hh]**2*M - start_frame[vv, hh]**2*B - mid_frame[vv, hh]**2*A \
                + mid_frame[vv, hh]**2*B + stop_frame[vv, hh]**2*A - stop_frame[vv, hh]**2*M) \
                / (start_frame[vv, hh]**2*mid_frame[vv, hh] - start_frame[vv, hh]**2*stop_frame[vv, hh] \
                - start_frame[vv, hh]*mid_frame[vv, hh]**2 + start_frame[vv, hh]*stop_frame[vv, hh]**2 \
                + mid_frame[vv, hh]**2*stop_frame[vv, hh] - mid_frame[vv, hh]*stop_frame[vv, hh]**2)
            c_nuc[vv, hh] = (start_frame[vv, hh]**2*mid_frame[vv, hh]*B - start_frame[vv, hh]**2 \
                * stop_frame[vv, hh]*M - start_frame[vv, hh]*mid_frame[vv, hh]**2*B + start_frame[vv, hh] \
                * stop_frame[vv, hh]**2*M + mid_frame[vv, hh]**2*stop_frame[vv, hh]*A - mid_frame[vv, hh] \
                * stop_frame[vv, hh]**2*A)/(start_frame[vv, hh]**2*mid_frame[vv, hh] - start_frame[vv, hh]**2 \
                * stop_frame[vv, hh] - start_frame[vv, hh]*mid_frame[vv, hh]**2 + start_frame[vv, hh] \
                * stop_frame[vv, hh]**2 + mid_frame[vv, hh]**2*stop_frame[vv, hh] - mid_frame[vv, hh]*stop_frame[vv, hh]**2)
    return a_nuc, b_nuc, c_nuc

def nuc_quad(data, coefs):
    out = np.zeros(data.shape)
    a_nuc, b_nuc, c_nuc = coefs
    for i in range(len(data)):
        out[i] = a_nuc*data[i]**2 + b_nuc*data[i] + c_nuc
    return out

def coeff_calc(frames, temps, points=2, quad=0):
    # Calculate the koefficients for each pixel to perform NUC
    start_frame = frames[0]
    stop_frame = frames[-1]
    A = np.average(start_frame)
    B = np.average(stop_frame)
    if points == 1:  # 1 point NUC
        b_nuc = A - frames[0]
        return [b_nuc]
    elif points == 2:  # 2 point NUC
        point_temps = (temps[0], temps[-1])
        print("Point tempers: \n", point_temps)
        k_avg = (-1) * (A - B) / (point_temps[1] - point_temps[0])
        b_avg = (-1) * (point_temps[0] * B - point_temps[1] * A) / (point_temps[1] - point_temps[0])
        k_pix = (-1) * (start_frame - stop_frame) / (point_temps[1] - point_temps[0])
        b_pix = (-1) * (point_temps[0] * stop_frame - point_temps[1] * start_frame) / (point_temps[1] - point_temps[0])
        k_nuc = k_avg / k_pix
        b_nuc = (-1) * (k_avg * b_pix) / k_pix + b_avg
        return k_nuc, b_nuc
    elif points == 3 and quad == 1:  # 3 point quadratic NUC
        return coeff_calc_quad(frames)
    elif points == 3 and quad == 0:  # 3 point bilinear NUC
        n = len(frames) // 2 if len(frames) % 2 else len(frames) // 2 - 1
        point_temps = (temps[0], temps[n], temps[-1])
        print("Point tempers: \n", point_temps)
        mid_frame = frames[n]
        M = np.average(mid_frame)  # Optional average value at the middle of Temps
        #k_avg = (-1) * (A - B) / (point_temps[2] - point_temps[0])
        #b_avg = (-1) * (point_temps[0] * B - point_temps[2] * A) / (point_temps[2] - point_temps[0])
        k_avg1 = (-1) * (A - M) / (point_temps[1] - point_temps[0])
        b_avg1 = (-1) * (point_temps[0] * M - point_temps[1] * A) / (point_temps[1] - point_temps[0])
        k_avg2 = (-1) * (M - B) / (point_temps[2] - point_temps[1])
        b_avg2 = (-1) * (point_temps[1] * B - point_temps[2] * M) / (point_temps[2] - point_temps[1])
        k_pix1 = (-1) * (start_frame - mid_frame) / (point_temps[1] - point_temps[0])
        b_pix1 = (-1) * (point_temps[0] * mid_frame - point_temps[1] * start_frame) / (point_temps[1] - point_temps[0])
        k_pix2 = (-1) * (mid_frame - stop_frame) / (point_temps[2] - point_temps[1])
        b_pix2 = (-1) * (point_temps[1] * stop_frame - point_temps[2] * mid_frame) / (point_temps[2] - point_temps[1])
        #k_nuc1 = k_avg / k_pix1
        #b_nuc1 = (-1) * (k_avg * b_pix1) / k_pix1 + b_avg
        #k_nuc2 = k_avg / k_pix2
        #b_nuc2 = (-1) * (k_avg * b_pix2) / k_pix2 + b_avg
        k_nuc1 = k_avg1 / k_pix1
        b_nuc1 = (-1) * (k_avg1 * b_pix1) / k_pix1 + b_avg1
        k_nuc2 = k_avg2 / k_pix2
        b_nuc2 = (-1) * (k_avg2 * b_pix2) / k_pix2 + b_avg2
        return k_nuc1, k_nuc2, b_nuc1, b_nuc2


def nuc(data, coefs):
    # Perform 1-3 point NUC
    out_unif = np.zeros((data.shape))
    f, pix_v, pix_h = data.shape
    it = np.arange(f)
    if len(coefs) == 1: # 1 point NUC
        b_nuc = coefs[0]
        for i in it:
            out_unif[i] = data[i] + b_nuc
    elif len(coefs) == 2:  # 2 point NUC
        k_nuc, b_nuc = coefs
        for i in it:
            out_unif[i] = k_nuc * data[i] + b_nuc
    elif len(coefs) == 3:  # 3 point quadratic NUC
        return nuc_quad(data, coefs)
    elif len(coefs) == 4:  # 3 point bilinear NUC
        k_nuc1, k_nuc2, b_nuc1, b_nuc2 = coefs
        
        row = np.arange(pix_v)
        column = np.arange(pix_h)
        mid_frame = data[f // 2 if f % 2 else f // 2 - 1]
        
        for i in it:
            for r in row:
                for c in column:
                    if (data[i][r][c] <= mid_frame[r][c]):
                        out_unif[i][r][c] = k_nuc1[r][c] * data[i][r][c] + b_nuc1[r][c]
                    elif (data[i][r][c] > mid_frame[r][c]):
                        out_unif[i][r][c] = k_nuc2[r][c] * data[i][r][c] + b_nuc2[r][c]
    
    return out_unif


def coeff_linear_temp(frames, temps):
    A = np.average(frames[0])
    B = np.average(frames[-1])
    n = len(frames) // 2 if len(frames) % 2 else len(frames) // 2 - 1
    M = np.average(frames[n])
    T0, T1, T2 = temps[0], temps[n], temps[-1]
    
    # sp.solve((a*A**2+b*A+c-T0, a*M**2+b*M+c-T1, a*B**2+b*B+c-T2), (a, b, c))
    lt_a = (A*T1 - A*T2 + B*T0 - B*T1 - M*T0 + M*T2)/(A**2*B - A**2*M - A*B**2 + A*M**2 + B**2*M - B*M**2)
    lt_b = (-A**2*T1 + A**2*T2 - B**2*T0 + B**2*T1 + M**2*T0 - M**2*T2)/(A**2*B - A**2*M - A*B**2 + A*M**2 + B**2*M - B*M**2)
    lt_c = (A**2*B*T1 - A**2*M*T2 - A*B**2*T1 + A*M**2*T2 + B**2*M*T0 - B*M**2*T0)/(A**2*B - A**2*M - A*B**2 + A*M**2 + B**2*M - B*M**2)
    
    return lt_a, lt_b, lt_c


def linear_temp(frames, coefs):
    out = np.zeros(frames.shape)
    for f in range(len(frames)):
        out[f] = coefs[0]*frames[f]**2 + coefs[1]*frames[f] + coefs[2]
    return out

