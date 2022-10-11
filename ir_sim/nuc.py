import sys
import numpy as np

from . import sys_param as sp


def coeff_calc(frames, temps=None, average=False, points=2, quad=0):
    # Calculate the koefficients for each pixel to perform NUC
    if points == 1:  # 1 point NUC
        return [coeff_calc_1point(frames)]
    elif points == 2:  # 2 point NUC
        return coeff_calc_linear(frames, temps, average)
    elif points == 3 and quad == 1:  # 3 point quadratic NUC
        return coeff_calc_quad(frames, temps, average)
    elif points == 3 and quad == 0:  # 3 point linear NUC
        return coeff_calc_linear_2piece(frames, temps, average)

def coeff_calc_1point(frames):
    f0 = frames[0]
    V0 = np.average(f0)
    return V0 - f0

def coeff_calc_linear(frames, temps=None, average=False):
    f0 = frames[0]
    f1 = frames[-1]
    if temps is None:
        V0 = np.average(f0)
        V1 = np.average(f1)
    else:
        V0 = temps[0]
        V1 = temps[-1]
        if average:
            f0 = np.average(f0)
            f1 = np.average(f1)
    
    nuc_a = (V0 - V1) / (f0 - f1)
    nuc_b = (V1*f0 - V0*f1) / (f0 - f1)
    return nuc_a, nuc_b

def coeff_calc_linear_2piece(frames, temps=None, average=False):
    f0 = frames[0]
    f2 = frames[-1]
    n = len(frames) // 2
    f1 = frames[n]
    
    if temps is None:
        V0 = np.average(f0)
        V1 = np.average(f1)
        V2 = np.average(f2)
    else:
        V0 = temps[0]
        V1 = temps[n]
        V2 = temps[-1]
        if average:
            f0 = np.average(f0)
            f1 = np.average(f1)
            f2 = np.average(f2)
    
    nuc1_a = (V0 - V1) / (f0 - f1)
    nuc1_b = (V1*f0 - V0*f1) / (f0 - f1)
    nuc2_a = (V1 - V2) / (f1 - f2)
    nuc2_b = (V2*f1 - V1*f2) / (f1 - f2)
    return nuc1_a, nuc2_a, nuc1_b, nuc2_b

def coeff_calc_quad(frames, temps=None, average=False):
    f0 = frames[0]
    f2 = frames[-1]
    n = len(frames) // 2
    f1 = frames[n]
    
    if temps is None:
        V0 = np.average(f0)
        V1 = np.average(f1)
        V2 = np.average(f2)
    else:
        V0 = temps[0]
        V1 = temps[n]
        V2 = temps[-1]
        if average:
            f0 = np.average(f0)
            f1 = np.average(f1)
            f2 = np.average(f2)
    
    a_nuc = (-f0*V1 + f0*V2 + f1*V0 - f1*V2 - f2*V0 + f2*V1) / (f0**2*f1 - f0**2*f2 - f0*f1**2 + f0*f2**2 + f1**2*f2 - f1*f2**2)
    b_nuc = (f0**2*V1 - f0**2*V2 - f1**2*V0 + f1**2*V2 + f2**2*V0 - f2**2*V1) / (f0**2*f1 - f0**2*f2 - f0*f1**2 + f0*f2**2 + f1**2*f2 - f1*f2**2)
    c_nuc = (f0**2*f1*V2 - f0**2 * f2*V1 - f0*f1**2*V2 + f0*f2**2*V1 + f1**2*f2*V0 - f1*f2**2*V0) / (f0**2*f1 - f0**2 * f2 - f0*f1**2 + f0 * f2**2 + f1**2*f2 - f1*f2**2)
    return a_nuc, b_nuc, c_nuc

def nuc(data, coefs):
    # Perform 1-3 point NUC
    if len(coefs) == 1: # 1 point NUC
        return nuc_1point(data, coefs[0])
    elif len(coefs) == 2:  # 2 point NUC
        return nuc_linear(data, coefs)
    elif len(coefs) == 3:  # 3 point quadratic NUC
        return nuc_quad(data, coefs)
    elif len(coefs) == 4:  # 3 point linear NUC
        return nuc_linear_2piece(data, coefs)

def nuc_1point(data, b):
    out_unif = np.zeros((data.shape))
    f, pix_v, pix_h = data.shape
    for i in np.arange(f):
        out_unif[i] = data[i] + b
    return out_unif

def nuc_linear(data, coefs):
    k_nuc, b_nuc = coefs
    out_unif = np.zeros((data.shape))
    f, pix_v, pix_h = data.shape 
    for i in np.arange(f):
        out_unif[i] = k_nuc * data[i] + b_nuc
    return out_unif

def nuc_linear_2piece(data, coefs):
    k_nuc1, k_nuc2, b_nuc1, b_nuc2 = coefs
    x_intersect = (b_nuc2 - b_nuc1) / (k_nuc1 - k_nuc2)
    out_unif = np.zeros((data.shape))
    f, pix_v, pix_h = data.shape
    for i in np.arange(f):
        out_unif[i] = np.where(data[i] <= x_intersect, k_nuc1*data[i]+b_nuc1, k_nuc2*data[i]+b_nuc2)
    return out_unif

def nuc_quad(data, coefs):
    out = np.zeros(data.shape)
    a_nuc, b_nuc, c_nuc = coefs
    for i in range(len(data)):
        out[i] = a_nuc*data[i]**2 + b_nuc*data[i] + c_nuc
    return out

