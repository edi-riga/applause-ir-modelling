import numpy as np
from numpy import exp
from scipy.optimize import fsolve
from scipy.constants import k
from time import localtime, strftime

from . import sys_param as sp
from .tolerance import generate_arrays as tol_arrays
from . import blackbody as bb

class FrameGen:
    def __init__(self, **kwargs):
        self.sensor = kwargs['sensor'] if 'sensor' in kwargs.keys() else (sp.pix_v_all, sp.pix_h_all, sp.skimming_pix, sp.boundary_pix)
        self.param = kwargs['param'] if 'param' in kwargs.keys() else (sp.Ib, sp.Ea, sp.T_sa, sp.int_time)
        self.opamp = kwargs['opamp'] if 'opamp' in kwargs.keys() else (sp.R1, sp.R2, sp.R3, sp.C)
        self.skimmingcolumn = kwargs['skimmingcolumn'] if 'skimmingcolumn' in kwargs.keys() else False
        
        if 'tol' in kwargs.keys():
            self.tol = kwargs['tol']
        else:
            self.tol = tol_arrays(size=(self.sensor[0], self.sensor[1]), Ea = self.param[1], T_s = self.param[2])
    
    def is_pixel_skimming(self, r, col):
        pix_v_all, pix_h_all, s_pix, _ = self.sensor
        return ((r==0) or (r<=(s_pix-1)) or (r==(pix_v_all-1)) or (r >= (pix_v_all - s_pix)) or (col == 0) or \
                (col <= (s_pix-1)) or (col >= (pix_h_all - s_pix)) or  (col == (pix_h_all -1 )))
    
    def is_pixel_boundary(self, r, col):
        pix_v_all, pix_h_all, s_pix, b_pix = self.sensor
        return ((r >= s_pix) and (r < (s_pix + b_pix))) or ((r >= (pix_v_all - s_pix - b_pix)) and \
            (r < (pix_v_all - s_pix))) or ( (r >= (s_pix + b_pix)) and (r < (pix_v_all - s_pix - b_pix)) and \
            (col >= s_pix) and (col < (s_pix + b_pix)) ) or ( (r >= (s_pix + b_pix)) and \
            (r < (pix_v_all - s_pix - b_pix)) and (col >= (pix_h_all - s_pix - b_pix)) and \
            (col < (pix_h_all - s_pix)))

    def is_pixel_active(self, r, col):
        pix_v_all, pix_h_all, s_pix, b_pix = self.sensor
        return (((r >= (s_pix + b_pix)) and (r < (pix_v_all - s_pix - b_pix))) and \
            ((col >= (s_pix + b_pix)) and (col < (pix_h_all - s_pix - b_pix))))

    def solve_pixel(self, r, col, Q):
        """Calculates pixel voltage integrated over the set integration time"""
        Rta, g_th, c_th, R0, tau = self.tol
        I, Ea, T_s, itime = self.param
        return fsolve(lambda Va: Va - I*R0[r][col] * exp(Ea / (k * T_s + k*((I*Va + Q) / \
            g_th[r][col])*(1+(tau[r][col] / itime) * (exp(-itime/tau[r][col]) - 1)  ) )), 0)
    
    def solve_pixel_inst(self, r, col):
        """Calculates the instantaneous value of a pixel voltage at the end of integration time"""
        Rta, g_th, c_th, R0, tau = self.tol
        I, Ea, T_s, itime = self.param
        return fsolve(lambda Vs: Vs - I * R0[r][col] * exp(Ea / ( k * (T_s + (I*Vs/g_th[r][col]) * \
            (1 - exp(-itime/tau[r][col]))))), 0)
    
    def pixel_values(self, i, r, col):
        """Calculates output voltage of each active pixels
        at given bias current, sensor temperature and IR power impinged"""
        pix_v_all, pix_h_all, s_pix, b_pix = self.sensor
        if self.is_pixel_skimming(r, col):
            self.V_all[i][r][col] = self.solve_pixel(r, col, 0)
            self.V_skim [i][r][col] = self.solve_pixel_inst(r, col)
        elif self.is_pixel_boundary(r, col):
            #Q = self.Pcam[0]
            Q = self.Pcam[i]
            self.V_all[i][r][col] =  self.solve_pixel(r, col, Q)
        elif self.is_pixel_active(r, col):
            #Q = self.P[i][r - s_pix - b_pix][col - s_pix - b_pix] + self.Pcam[0]
            Q = self.P[i][r - s_pix - b_pix][col - s_pix - b_pix] + self.Pcam[i]
            self.V_all[i][r][col] =  self.solve_pixel(r, col, Q)
    
    def run_pixels(self, P, Pcam):
        pix_v_all, pix_h_all, s_pix, b_pix = self.sensor
        self.P = P
        if not isinstance(Pcam, np.ndarray) or len(Pcam) == 1:
            self.Pcam = Pcam + np.zeros(len(P))
        else:
            self.Pcam = Pcam
        self.V_all   = np.ones((len(P), pix_v_all, pix_h_all))
        self.V_skim  = np.ones((len(P), pix_v_all, pix_h_all))
        
        row = np.arange(pix_v_all)
        column = np.arange(pix_h_all)
        for i in np.arange(len(P)):
            nr = i+1
            print("Processing frame Nr %s" %nr)
            print(strftime("%H:%M:%S", localtime()))
            for r in row:
                for col in column:
                    self.pixel_values(i, r, col)
       
    def roic_function(self, i, r, col):
        pix_v_all, pix_h_all, s_pix, b_pix = self.sensor
        R1, R2, R3, C = self.opamp
        V_all = self.V_all
        V_skim = self.V_skim
        
        if self.skimmingcolumn:
            # Skimming column is chosen
            # TO DO: PARASITIC DECREASE OF NON-INVERTING INPUT IMPEDANCE,
            # BECAUSE SKIMMING PIXELS ARE SIMULTANEOUSLY CONNECTED TO ALL INTEGRATORS IN THE ROW.
            if s_pix == 1:
                # Average value of skimming in the current row
                skimming_average_int = np.average( [V_all[i][r+s_pix][0], V_all[i][r+s_pix][pix_h_all - 1]] )
                skimming_average = np.average([V_skim[i][r+s_pix][0], V_skim[i][r+s_pix][pix_h_all - 1]])
                self.V_out[i][r][col] = (1/(R1*C)) * ( (R3 /(R2+R3)) * skimming_average_int \
                    - V_all[i][r+s_pix][col+s_pix] ) + (R3 /(R2+R3) * skimming_average)
            else:
                pass # TO DO: Here should be defined the ROIC function if sensor has more than 1 ring of skimming pixels
            # Integrators' analog output saturation:
            if self.V_out[i][r][col] < 0:
                self.V_out[i][r][col] = 0
            elif self.V_out[i][r][col] >= 3.2:
                self.V_out[i][r][col] = 3.2
        else:
            # Skimming row is chosen
            # TO DO: Skimming pixels become HOT!!!
            if s_pix == 1:
                # Average value of skimming in the current row
                skimming_average_int = np.average([V_all[i][0][col+s_pix], V_all[i][pix_v_all - 1][col+s_pix]])
                skimming_average = np.average([V_skim[i][0][col+s_pix], V_skim[i][pix_v_all - 1][col+s_pix]])
                self.V_out[i][r][col] = (1 / (R1 * C)) * ((R3 /(R2+R3)) * skimming_average_int\
                    - V_all[i][r+s_pix][col+s_pix])+(R3 /(R2+R3)*skimming_average)
            else:
                pass # TO DO: Here should be defined the ROIC function if sensor has more than 1 ring of skimming pixels
            
            if self.V_out[i][r][col] < 0:
                self.V_out[i][r][col] = 0
            elif self.V_out[i][r][col] >= 3.2:
                self.V_out[i][r][col] = 3.2
            # TO DO: If external voltage is used as the reference of integrator (NON-INVERTING INPUT)
            # elif args.externalvoltage:
            #   # External voltage as reference
            #   V_out[i][r+s_pix][col+s_pix] = (1 / (R1 * C)) * (V_ext*itime - V_all[i][r+s_pix][col+s_pix])
    
    def run_frames(self, P, Pcam):
        self.run_pixels(P, Pcam)
        self.run_roic()
        return self.V_out
    
    def run_roic(self):
        pix_v_all, pix_h_all, s_pix, b_pix = self.sensor
        self.V_out = np.ones((len(self.P), pix_v_all-s_pix*2, pix_h_all-s_pix*2))
        row_for_out = np.arange(pix_v_all-s_pix*2)
        col_for_out = np.arange(pix_h_all-s_pix*2)
        for i in np.arange(len(self.P)):
            for r in row_for_out:
                for col in col_for_out:
                    self.roic_function(i, r, col)
    
    def filter_actives(self, frames):
        """Returns the active pixels of a complete pixel array"""
        pix_v_all, pix_h_all, s_pix, b_pix = self.sensor
        pix_v = pix_v_all - s_pix*2 - b_pix*2
        pix_h = pix_h_all - s_pix*2 - b_pix*2
        row_im = np.arange(pix_v)
        col_im = np.arange(pix_h)
        out = np.ones((len(frames), pix_v, pix_h))
        for i in np.arange(len(frames)):
            for r in row_im:
                for col in col_im:
                    out[i][r][col] = frames[i][r + s_pix + b_pix - 1][col + s_pix + b_pix - 1]
        return out
    
    def boundary_average(self, frames):
        """
        """
        pix_v_all, pix_h_all, s_pix, b_pix = self.sensor
        bavg = np.zeros(len(frames))
        for f in range(len(frames)):
            b_hor = np.concatenate([frames[f,:,:s_pix], frames[f,:,-s_pix:]])
            b_ver = np.concatenate([frames[f,:s_pix,:], frames[f,-s_pix:,:]])
            bavg[f] = np.average(np.concatenate([b_hor.reshape(b_hor.size), b_ver.reshape(b_ver.size)]))
        return bavg


def convert(img, k=sp.adc_coef):
    return (img * k).astype(np.uint8)

