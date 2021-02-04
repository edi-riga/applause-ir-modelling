import numpy as np
from numpy import exp
from scipy.optimize import fsolve
from scipy.constants import k
from time import localtime, strftime

from . import sys_param as sp
from .tolerance import generate_arrays as tol_arrays
from . import blackbody as bb

class FrameGen:
    def __init__(self, pix_h=sp.pix_h, pix_v=sp.pix_v, pix_skimming = sp.skimming_pix,
                 pix_boundary=sp.boundary_pix, skimmingcolumn=False, Ib=sp.Ib, Ea=sp.Ea,
                 T_sa=sp.T_sa, int_time=sp.int_time, R1=sp.R1, R2=sp.R2, R3=sp.R3, C=sp.C,
                 R_tol=sp.R_tol, g_tol=sp.g_tol, c_tol=sp.c_tol, R_tai = sp.R_ta_i, gi=sp.g_ini,
                 ci=sp.c_ini, seed=None, lambd=(sp.lambd1, sp.lambd2), Phi=(sp.Phi_r, sp.Phi_s),
                 A_sens=sp.A_sens, Omega=sp.Omega, fl=sp.fl, pitch=sp.pitch, max_analog=sp.max_analog):
        
        # Pixels
        self._pix_h = pix_h
        self._pix_v = pix_v
        self._pix_boundary = pix_boundary
        self._pix_skimming = pix_skimming
        
        self._update_pix_v()
        self._update_pix_h()
        self._update_masks()
        
        # Other parameters
        self.Ib = Ib
        self.Ea = Ea
        self.T_sa = T_sa
        self.int_time = int_time
        
        self.R1 = R1
        self.R2 = R2
        self.R3 = R3
        self.C = C
        
        self.skimmingcolumn = skimmingcolumn
        
        # Tolerance
        self.R_tol = R_tol
        self.g_tol = g_tol
        self.c_tol = c_tol
        self.R_tai = R_tai
        self.gi = gi
        self.ci = ci
        self.seed = seed
        
        # Black body
        self.lambd = lambd
        self.Phi = Phi
        self.A_sens = A_sens
        self.Omega = Omega
        self.fl = fl
        self.pitch = pitch
        
        self.max_analog = max_analog
    
    
    @property
    def pix_h(self):
        return self._pix_h
    
    @pix_h.setter
    def pix_h(self, val):
        self._pix_h = val
        self._update_pix_h()
        self._update_masks()
    
    @property
    def pix_v(self):
        return self._pix_v
    
    @pix_v.setter
    def pix_v(self, val):
        self._pix_v = val
        self._update_pix_v()
        self._update_masks()
    
    @property
    def pix_boundary(self):
        return self._pix_boundary
    
    @pix_boundary.setter
    def pix_boundary(self, val):
        self._pix_boundary = val
        self._update_pix_v()
        self._update_pix_h()
        self._update_masks()
    
    @property
    def pix_skimming(self):
        return self._pix_skimming
    
    @pix_skimming.setter
    def pix_skimming(self, val):
        self._pix_skimming = val
        self._update_pix_v()
        self._update_pix_h()
        self._update_masks()
    
    @property
    def pix_h_all(self):
        return self._pix_h_all
    
    @property
    def pix_v_all(self):
        return self._pix_v_all
    
    def _update_pix_v(self):
        self._pix_v_all = self._pix_v + self._pix_boundary*2 + self._pix_skimming*2
    
    def _update_pix_h(self):
        self._pix_h_all = self._pix_h + self._pix_boundary*2 + self._pix_skimming*2
    
    def _update_masks(self):
        pix_bs = self._pix_boundary + self._pix_skimming
        mask_active = np.zeros((self._pix_v_all, self._pix_h_all))
        mask_active[pix_bs:self._pix_v_all-pix_bs, pix_bs:self._pix_h_all-pix_bs] = np.ones((self._pix_v, self._pix_h))
        mask_skimming = np.ones((self._pix_v_all, self._pix_h_all))
        mask_skimming[self.pix_skimming:self._pix_v_all-self.pix_skimming, \
                      self.pix_skimming:self._pix_h_all-self.pix_skimming] = np.zeros((self._pix_v + 2*self._pix_boundary, \
                                                                                       self._pix_h + 2*self._pix_boundary))
        self.mask_active = mask_active.astype(np.bool)
        self.mask_skimming = mask_skimming.astype(np.bool)
        self.mask_boundary = np.logical_and(np.logical_not(mask_active), np.logical_not(mask_skimming))
    
    def _generate_tol_arrays(self):
        self.Rta, self.g, \
        self.c, self.R0, self.tau = tol_arrays(self.R_tol, self.g_tol, self.c_tol, (self.pix_v_all,
                                               self.pix_h_all), self.R_tai, self.gi, self.ci,
                                               self.Ea, self.T_sa, self.seed) 
    
    def is_pixel_skimming(self, r, col):
        return self.mask_skimming[r, col]
    
    def is_pixel_boundary(self, r, col):
        return self.mask_boundary[r, col]
    
    def is_pixel_active(self, r, col):
        return self.mask_active[r, col]
    
    def solve_pixel(self, r, col, Q):
        """Calculates pixel voltage integrated over the set integration time"""
        return fsolve(lambda Va: Va - self.Ib*self.R0[r][col] * exp(self.Ea / (k * self.T_sa + k*((self.Ib*Va + Q) / \
            self.g[r][col])*(1+(self.tau[r][col] / self.int_time) * (exp(-self.int_time/self.tau[r][col]) - 1)  ) )), 0)
    
    def solve_pixel_inst(self, r, col):
        """Calculates the instantaneous value of a pixel voltage at the end of integration time"""
        return fsolve(lambda Vs: Vs - self.Ib * self.R0[r][col] * exp(self.Ea / ( k * (self.T_sa + (self.Ib*Vs/self.g[r][col]) * \
            (1 - exp(-self.int_time/self.tau[r][col]))))), 0)
    
    def pixel_values(self, i, r, col):
        """Calculates output voltage of each active pixels
        at given bias current, sensor temperature and IR power impinged"""
        if self.is_pixel_skimming(r, col):
            self.V_all[i][r][col] = self.solve_pixel(r, col, 0)
            self.V_skim [i][r][col] = self.solve_pixel_inst(r, col)
        elif self.is_pixel_boundary(r, col):
            Q = self.Pcam[i]
            self.V_all[i][r][col] =  self.solve_pixel(r, col, Q)
        elif self.is_pixel_active(r, col):
            Q = self.P[i][r - self.pix_skimming - self.pix_boundary][col - self.pix_skimming - self.pix_boundary] + self.Pcam[i]
            self.V_all[i][r][col] =  self.solve_pixel(r, col, Q)
    
    def run_pixels(self, P, Pcam):
        self._generate_tol_arrays()
        self.P = P
        if not isinstance(Pcam, np.ndarray) or len(Pcam) == 1:
            self.Pcam = Pcam + np.zeros(len(P))  # Constant camera temperature
        else:
            self.Pcam = Pcam
        self.V_all   = np.ones((len(P), self.pix_v_all, self.pix_h_all))
        self.V_skim  = np.ones((len(P), self.pix_v_all, self.pix_h_all))
        
        row = np.arange(self.pix_v_all)
        column = np.arange(self.pix_h_all)
        for i in np.arange(len(P)):
            print(f'[{strftime("%H:%M:%S", localtime())}] Processing frame {i+1}/{len(P)}')
            for r in row:
                for col in column:
                    self.pixel_values(i, r, col)
    
    def roic_function(self, i, r, col):
        if self.skimmingcolumn:
            # Skimming column is chosen
            # TO DO: PARASITIC DECREASE OF NON-INVERTING INPUT IMPEDANCE,
            # BECAUSE SKIMMING PIXELS ARE SIMULTANEOUSLY CONNECTED TO ALL INTEGRATORS IN THE ROW.
            if self.pix_skimming == 1:
                # Average value of skimming in the current row
                skimming_average_int = np.average( [self.V_all[i][r+self.pix_skimming][0], self.V_all[i][r+self.pix_skimming][self.pix_h_all - 1]] )
                skimming_average = np.average([self.V_skim[i][r+self.pix_skimming][0], self.V_skim[i][r+self.pix_skimming][self.pix_h_all - 1]])
                self.V_out[i][r][col] = (1/(self.R1*self.C)) * ( (self.R3 /(self.R2+self.R3)) * skimming_average_int \
                    - self.V_all[i][r+self.pix_skimming][col+self.pix_skimming] ) + (self.R3 /(self.R2+self.R3) * skimming_average)
            else:
                pass # TO DO: Here should be defined the ROIC function if sensor has more than 1 ring of skimming pixels
            # Integrators' analog output saturation:
            if self.V_out[i][r][col] < 0:
                self.V_out[i][r][col] = 0
            elif self.V_out[i][r][col] >= self.max_analog:
                self.V_out[i][r][col] = self.max_analog
        else:
            # Skimming row is chosen
            # TO DO: Skimming pixels become HOT!!!
            if self.pix_skimming == 1:
                # Average value of skimming in the current row
                skimming_average_int = np.average([self.V_all[i][0][col+self.pix_skimming], self.V_all[i][self.pix_v_all - 1][col+self.pix_skimming]])
                skimming_average = np.average([self.V_skim[i][0][col+self.pix_skimming], self.V_skim[i][self.pix_v_all - 1][col+self.pix_skimming]])
                self.V_out[i][r][col] = (1 / (self.R1 * self.C)) * ((self.R3 /(self.R2+self.R3)) * skimming_average_int\
                    - self.V_all[i][r+self.pix_skimming][col+self.pix_skimming])+(self.R3 /(self.R2+self.R3)*skimming_average)
            else:
                pass # TO DO: Here should be defined the ROIC function if sensor has more than 1 ring of skimming pixels
            
            if self.V_out[i][r][col] < 0:
                self.V_out[i][r][col] = 0
            elif self.V_out[i][r][col] >= self.max_analog:
                self.V_out[i][r][col] = self.max_analog
            # TO DO: If external voltage is used as the reference of integrator (NON-INVERTING INPUT)
            # elif args.externalvoltage:
            #   # External voltage as reference
            #   V_out[i][r+s_pix][col+s_pix] = (1 / (R1 * C)) * (V_ext*itime - V_all[i][r+s_pix][col+s_pix])
    
    def run_frames(self, T, Tcam):
        """
        
        Parameters
        ----------
        T: array
            Black body temperature array
        
        Tcam: array, float
            Camera body temperature or temperature array
        
        """
        if not isinstance(Tcam, np.ndarray) or len(Tcam) == 1:
            Tcam_ = Tcam + np.zeros(len(T))
        else:
            Tcam_ = Tcam
        
        Pcam = bb.integration(Tcam_, lambd=self.lambd, Phi=self.Phi, A_sens=self.A_sens, Omega=self.Omega)
        P = bb.integration(T, lambd=self.lambd, Phi=self.Phi, A_sens=self.A_sens, Omega=self.Omega)
        PD = bb.power_distribution_over_sens_area(P, size=(self.pix_v, self.pix_h), fl=self.fl, pitch=self.pitch)
        
        self.run_pixels(PD, Pcam)
        self.run_roic()
        return self.V_out
    
    def run_roic(self):
        self.V_out = np.ones((len(self.P), self.pix_v_all-self.pix_skimming*2, self.pix_h_all-self.pix_skimming*2))
        row_for_out = np.arange(self.pix_v_all-self.pix_skimming*2)
        col_for_out = np.arange(self.pix_h_all-self.pix_skimming*2)
        for i in np.arange(len(self.P)):
            for r in row_for_out:
                for col in col_for_out:
                    self.roic_function(i, r, col)
    
    def filter_actives(self, frames):
        """Returns the active pixels of a complete pixel array"""
        row_im = np.arange(self.pix_v)
        col_im = np.arange(self.pix_h)
        out = np.ones((len(frames), self.pix_v, self.pix_h))
        for i in np.arange(len(frames)):
            for r in row_im:
                for col in col_im:
                    out[i][r][col] = frames[i][r + self.pix_skimming + self.pix_boundary - 1][col + self.pix_skimming + self.pix_boundary - 1]
        return out
    
    def boundary_average(self, frames):
        """
        """
        bavg = np.zeros(len(frames))
        for f in range(len(frames)):
            b_hor = np.concatenate([frames[f,:,:self.pix_skimming], frames[f,:,-self.pix_skimming:]])
            b_ver = np.concatenate([frames[f,:self.pix_skimming,:], frames[f,-self.pix_skimming:,:]])
            bavg[f] = np.average(np.concatenate([b_hor.reshape(b_hor.size), b_ver.reshape(b_ver.size)]))
        return bavg


def convert(img, k=sp.adc_coef):
    return (img * k).astype(np.uint8)

