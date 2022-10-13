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
                 R_tol=sp.R_tol, g_tol=sp.g_tol, c_tol=sp.c_tol, R_tai=sp.R_ta_i, gi=sp.g_ini,
                 ci=sp.c_ini, seed=None, lambd=(sp.lambd1, sp.lambd2), Phi=(sp.Phi_r, sp.Phi_s),
                 A_sens=sp.A_sens, Omega=sp.Omega, fl=sp.fl, pitch=sp.pitch, max_analog=sp.max_analog,
                 Vref=None):
        """
        Parameters
        ----------
        pix_h:
            Horizontal sensor resolution of the sensor's active area.
        
        pix_v:
            Vertical resolution of the sensor's active area.
        
        pix_skimming: integer
            Number of skimming pixel rings.
        
        pix_boundary: integer
            Number of boundary pixel rings.
        
        skimmingcolumn: boolean, default: False
            Whether to use skimming column. Skimming row is used when set to False.
        
        Ib:
            
        Ea:
            
        T_sa:
            
        int_time:
            
        R1:
            
        R2:
            
        R3:
            
        C:
            
        max_analog:
            
        Vref:
            
        R_tol, g_tol, c_tol, R_tai, gi, ci, seed:
            See help(ir_sim.tolerance.generate_arrays)
        
        lambd, Phi, A_sens, Omega:
            See help(ir_sim.blackbody.integration)
        
        fl, pitch:
            See help(ir_sim.blackbody.power_distribution_over_sens_area)
        """
        
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
        self.Vref = Vref
    
    
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
        mask_skimming[self._pix_skimming:self._pix_v_all-self._pix_skimming, \
                      self._pix_skimming:self._pix_h_all-self._pix_skimming] = np.zeros((self._pix_v + 2*self._pix_boundary, \
                                                                                       self._pix_h + 2*self._pix_boundary))
        self.mask_active = mask_active.astype(np.bool)
        self.mask_skimming = mask_skimming.astype(np.bool)
        self.mask_boundary = np.logical_and(np.logical_not(mask_active), np.logical_not(mask_skimming))
    
    def _generate_tol_arrays(self):
        self.Rta, self.g, \
        self.c, self.R0, self.tau = tol_arrays(self.R_tol, self.g_tol, self.c_tol, (self._pix_v_all,
                                               self._pix_h_all), self.R_tai, self.gi, self.ci,
                                               self.Ea, self.T_sa, self.seed) 
    
    def is_pixel_skimming(self, r, col):
        return self.mask_skimming[r, col]
    
    def is_pixel_boundary(self, r, col):
        return self.mask_boundary[r, col]
    
    def is_pixel_active(self, r, col):
        return self.mask_active[r, col]
    
    def _solve_pixel(self, r, col, Q):
        """Calculates pixel voltage integrated over the set integration time"""
        return fsolve(lambda Va: Va - self.Ib*self.R0[r, col] * exp(self.Ea / (k * self.T_sa + k*((self.Ib*Va + Q) / \
            self.g[r, col])*(1+(self.tau[r, col] / self.int_time) * (exp(-self.int_time/self.tau[r, col]) - 1)  ) )), 0)
    
    def _solve_pixel_inst(self, r, col, Q):
        """Calculates the instantaneous value of a pixel voltage at the end of integration time"""
        return fsolve(lambda Vs: Vs - self.Ib * self.R0[r, col] * exp(self.Ea / ( k * (self.T_sa + ((self.Ib*Vs + Q)/self.g[r, col]) * \
            (1 - exp(-self.int_time/self.tau[r, col]))))), 0)
    
    def _run_pixels(self):
        """Calculates output voltage of each pixel at given bias
        current, sensor temperature and IR power impinged"""
        frames = len(self.Q)
        self.V_pix = np.zeros(self.Q.shape)
        self.V_pix_skim = np.zeros(self.Q.shape)
        
        skim_pix_coords = np.argwhere(self.mask_skimming)
        
        print("Calculating pixel voltages")
        for i in range(frames):
            print(f'[{strftime("%H:%M:%S", localtime())}] Processing frame {i+1}/{frames}')
            for r in range(self._pix_v_all):
                for col in range(self._pix_h_all):
                    self.V_pix[i, r, col] = self._solve_pixel(r, col, self.Q[i, r, col])
            for r, col in skim_pix_coords:
                self.V_pix_skim[i, r, col] = self._solve_pixel_inst(r, col, 0)
    
    def _roic_function(self, i, r, col):
        if not self.Vref:
            if self.skimmingcolumn:
                # Skimming column is chosen
                # TODO: PARASITIC DECREASE OF NON-INVERTING INPUT IMPEDANCE,
                # BECAUSE SKIMMING PIXELS ARE SIMULTANEOUSLY CONNECTED TO ALL INTEGRATORS IN THE ROW.
                if self._pix_skimming == 1:
                    # Average value of skimming in the current column
                    skimming_average_int = np.average([self.V_pix[i, r+self._pix_skimming, 0], self.V_pix[i, r+self._pix_skimming, self._pix_h_all - 1]])
                    skimming_average = np.average([self.V_pix_skim[i, r+self._pix_skimming, 0], self.V_pix_skim[i, r+self._pix_skimming, self._pix_h_all - 1]])
                else:
                    # TODO: ROIC function for sensor with more than 1 ring of skimming pixels
                    raise NotImplementedError('Only one ring of skimming pixels is supported.')
            
            else:
                # Skimming row is chosen
                # TODO: Skimming pixels become HOT!!!
                if self._pix_skimming == 1:
                    # Average value of skimming in the current row
                    skimming_average_int = np.average([self.V_pix[i, 0, col+self._pix_skimming], self.V_pix[i, self._pix_v_all - 1, col+self._pix_skimming]])
                    skimming_average = np.average([self.V_pix_skim[i, 0, col+self._pix_skimming], self.V_pix_skim[i, self._pix_v_all - 1, col+self._pix_skimming]])
                else:
                    # TODO: ROIC function for sensor with more than 1 ring of skimming pixels
                    raise NotImplementedError('Only one ring of skimming pixels is supported.')
            
            V_out = (1/(self.R1*self.C)) * ((self.R3 /(self.R2+self.R3)) * skimming_average_int \
                        - self.V_pix[i, r+self._pix_skimming, col+self._pix_skimming] ) + (self.R3 /(self.R2+self.R3) * skimming_average)
        
        else:
            # TODO: Voltage as reference
            # V_out[i][r+s_pix][col+s_pix] = (1 / (R1 * C)) * (V_ext*itime - V_all[i][r+s_pix][col+s_pix])
            raise NotImplementedError('Reference voltage is not supported.')
        
        # Integrators' analog output saturation:
        V_out = V_out if V_out >= 0 else 0
        V_out = V_out if V_out <= self.max_analog else self.max_analog
        
        self.V_roic[i, r+self._pix_skimming, col+self._pix_skimming] = V_out
    
    def _run_roic(self):
        self.V_roic = np.zeros(self.V_pix.shape)
        print("Calculating ROIC output")
        for i in np.arange(len(self.Q)):
            for r in np.arange(self._pix_v+2*self._pix_boundary):
                for col in np.arange(self._pix_h+2*self._pix_boundary):
                    self._roic_function(i, r, col)
    
    def run_frames(self, T, Tcam):
        """
        
        Parameters
        ----------
        T: array
            Black body temperature array
        
        Tcam: array, float
            Camera body temperature or temperature array
        
        """
        Tcam = Tcam + np.zeros(len(T))  # Tcam might not be an array
        
        Qcam1d = bb.integration(Tcam, lambd=self.lambd, Phi=self.Phi, A_sens=self.A_sens, Omega=self.Omega)
        Q0     = bb.integration(T,    lambd=self.lambd, Phi=self.Phi, A_sens=self.A_sens, Omega=self.Omega)
        Qbb    = bb.power_distribution_over_sens_area(Q0, size=(self._pix_v_all, self._pix_h_all), fl=self.fl, pitch=self.pitch)
        
        Qbb = np.where(self.mask_active, Qbb, 0)
        
        Qcam = np.zeros(Qbb.shape)
        active_and_boundary = np.logical_or(self.mask_boundary, self.mask_active)
        for f in range(len(Qcam)):
            Qcam[f] = np.where(active_and_boundary, Qcam[f]+Qcam1d[f], Qcam[f])
        
        self.Q = Qbb + Qcam
        
        self._generate_tol_arrays()
        self._run_pixels()
        self._run_roic()
        return self.V_roic
    
    def filter_boundary(self, frames=None):
        """
        """
        if frames is None:
            frames = self.V_roic
        mask = np.logical_not(self.mask_boundary)
        if (dims := len(frames.shape)) == 2:
            return np.ma.masked_array(frames, mask=mask)
        elif dims == 3:
            return np.ma.masked_array(frames, np.broadcast_to(mask, frames.shape))
    
    def filter_actives(self, frames=None):
        """
        """
        if frames is None:
            frames = self.V_roic
        if (dims := len(frames.shape)) == 2:
            return frames[self._pix_boundary+self._pix_skimming:-self._pix_boundary-self._pix_skimming,
                          self._pix_boundary+self._pix_skimming:-self._pix_boundary-self._pix_skimming]
        elif dims == 3:
            return frames[:,
                          self._pix_boundary+self._pix_skimming:-self._pix_boundary-self._pix_skimming,
                          self._pix_boundary+self._pix_skimming:-self._pix_boundary-self._pix_skimming]
    
    def boundary_average(self, frames=None):
        """
        """
        if frames is None:
            frames = self.V_roic
        if (dims := len(frames.shape)) == 2:
            return np.average(self.filter_boundary(frames))
        elif dims == 3:
            return np.average(self.filter_boundary(frames), axis=(1,2))


    def convert(self, img, k=sp.adc_coef):
        return (img * k).astype(np.uint8)

