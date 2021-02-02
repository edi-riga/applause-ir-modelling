# Camera temperature compensation test
import os

import numpy as np
import matplotlib.pyplot as plt

from ir_sim import frame_gen as g
from ir_sim import nuc
from ir_sim import dpd

# https://matplotlib.org/gallery/event_handling/image_slices_viewer.html
class IndexTracker:
    def __init__(self, ax, X, title='scroll to navigate', *args, **kwargs):
        self.ax = ax
        ax.set_title(title)

        self.X = X
        self.slices, _, _ = X.shape
        self.ind = 0

        self.im = ax.imshow(self.X[self.ind], *args, **kwargs)
        self.update()

    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.ind = (self.ind + 1) % self.slices
        else:
            self.ind = (self.ind - 1) % self.slices
        self.update()

    def update(self):
        self.im.set_data(self.X[self.ind])
        self.ax.set_ylabel('slice %s' % self.ind)
        self.im.axes.figure.canvas.draw()

class Tracker2D:
    def __init__(self, datas, fig, ax, lines):
        f, v, self.h = datas[0].shape
        self.pixels = v*self.h
        self.pixel = 0
        self.lines = lines
        self.datas = datas
        self.ax = ax
        self.fig = fig
    
    def on_scroll(self, event):
        print("%s %s" % (event.button, event.step))
        if event.button == 'up':
            self.pixel = (self.pixel + 1) % self.pixels
        else:
            self.pixel = (self.pixel - 1) % self.pixels
        
        for i in range(len(self.datas)):
            self.lines[i][0].set_ydata(self.datas[i][:, self.pixel // self.h, self.pixel % self.h])
        self.ax.set_ylabel(f'pix {self.pixel // self.h} {self.pixel % self.h}')
        self.fig.canvas.draw()


def imshow_i(data, title, *args, **kwargs):
    fig, ax = plt.subplots(1, 1)
    tracker = IndexTracker(ax, data, title, *args, **kwargs)
    fig.canvas.mpl_connect('scroll_event', tracker.on_scroll)
    return fig, ax, tracker

#############
# Parameters
#############

R_tol = g_tol = c_tol = 1e-5
T = np.linspace(300, 400, 10)
pix_v = 100
pix_h = 120
seed = 123

# C may be 0.8 (lowest gain), 0.4, 0.2 and 0.1 pF (highest gain)
C = 0.4e-12

gen_params = {
    'R_tol': R_tol,
    'g_tol': g_tol,
    'c_tol': c_tol,
    'pix_v': pix_v,
    'pix_h': pix_h,
    'seed': seed,
    'C': C
}

try:
    frames_all_up = np.loadtxt('data_test2/frames_up.txt').reshape(T.size, pix_v_all-sp.skimming_pix*2, pix_h_all-sp.skimming_pix*2)
    frames_all_down = np.loadtxt('data_test2/frames_down.txt').reshape(T.size, pix_v_all-sp.skimming_pix*2, pix_h_all-sp.skimming_pix*2)
    frames_all_const1 = np.loadtxt('data_test2/frames_const1.txt').reshape(T.size, pix_v_all-sp.skimming_pix*2, pix_h_all-sp.skimming_pix*2)
    frames_all_const2 = np.loadtxt('data_test2/frames_const2.txt').reshape(T.size, pix_v_all-sp.skimming_pix*2, pix_h_all-sp.skimming_pix*2)
    frames_all_const3 = np.loadtxt('data_test2/frames_const3.txt').reshape(T.size, pix_v_all-sp.skimming_pix*2, pix_h_all-sp.skimming_pix*2)
    gg = g.FrameGen(**gen_params)
except:
    Tcam_up = T
    Tcam_down = T[::-1]
    Tcam_const1 = Tcam_up[0] + np.zeros(Tcam_up.shape)
    Tcam_const2 = Tcam_up[-1] + np.zeros(Tcam_up.shape)
    Tcam_const3 = Tcam_up[len(Tcam_up) // 2] + np.zeros(Tcam_up.shape)
    
    gg = g.FrameGen(**gen_params)
    
    print('--- Rising camera temperature ---')
    frames_all_up = gg.run_frames(T, Tcam_up)
    print('--- Falling camera temperature ---')
    frames_all_down = gg.run_frames(T, Tcam_down)
    print('--- Constant camera temperature 1 ---')
    frames_all_const1 = gg.run_frames(T, Tcam_const1)
    print('--- Constant camera temperature 2 ---')
    frames_all_const2 = gg.run_frames(T, Tcam_const2)
    print('--- Constant camera temperature 3 ---')
    frames_all_const3 = gg.run_frames(T, Tcam_const3)
    
    try:
        os.mkdir('data_test2')
    except:
        pass
    
    with open('data_test2/frames_up.txt', 'w') as outf:
        for f in frames_all_up:
            np.savetxt(outf, f)
    
    with open('data_test2/frames_down.txt', 'w') as outf:
        for f in frames_all_down:
            np.savetxt(outf, f)
    
    with open('data_test2/frames_const1.txt', 'w') as outf:
        for f in frames_all_const1:
            np.savetxt(outf, f)
    
    with open('data_test2/frames_const2.txt', 'w') as outf:
        for f in frames_all_const2:
            np.savetxt(outf, f)
    
    with open('data_test2/frames_const3.txt', 'w') as outf:
        for f in frames_all_const3:
            np.savetxt(outf, f)
    

frames_up = gg.filter_actives(frames_all_up)
frames_down = gg.filter_actives(frames_all_down)
frames_const1 = gg.filter_actives(frames_all_const1)
frames_const2 = gg.filter_actives(frames_all_const2)
frames_const3 = gg.filter_actives(frames_all_const3)

#############
# Correction
#############

# non-uniformity correction coefficients calculated for one of the constant camera temperatures
dp = dpd.dpd(frames_const1)
frames_const1 = dpd.replace_dead(frames_const1, dp)
coefs = nuc.coeff_calc(frames_const1, T, points=3, quad=1)
corr_const1 = nuc.nuc(frames_const1, coefs)
corr_const1 = dpd.replace_dead(corr_const1, dpd.dpd(corr_const1))

# non-uniformity correction applied to all cases
corr_up = nuc.nuc(frames_up, coefs)
corr_down = nuc.nuc(frames_down, coefs)
corr_const2 = nuc.nuc(frames_const2, coefs)
corr_const3 = nuc.nuc(frames_const3, coefs)

# boundary pixel average values
bavg_const1 = gg.boundary_average(frames_all_const1)
bavg_const2 = gg.boundary_average(frames_all_const2)
bavg_const3 = gg.boundary_average(frames_all_const3)
bavg_up = gg.boundary_average(frames_all_up)
bavg_down = gg.boundary_average(frames_all_down)

# effect of the camera temperature is compensated by subtracting boundary pixel average
def sub_bavg(frames, bavg, bavg0):
    out = np.zeros(frames.shape)
    for f in range(len(frames)):
        out[f] = frames[f] - (bavg[f] - bavg0[f])
    return out

fixed_const1 = sub_bavg(corr_const1, bavg_const1, bavg_const1)
fixed_const2 = sub_bavg(corr_const2, bavg_const2, bavg_const1)
fixed_const3 = sub_bavg(corr_const3, bavg_const3, bavg_const1)
fixed_up = sub_bavg(corr_up, bavg_up, bavg_const1)
fixed_down = sub_bavg(corr_down, bavg_down, bavg_const1)


##########
# Figures
##########

fig, ax = plt.subplots()
line1 = ax.plot(corr_const1[:,0,0], label='const1')
line2 = ax.plot(corr_const2[:,0,0], label='const2')
line3 = ax.plot(corr_const3[:,0,0], label='const3')
line4 = ax.plot(corr_up[:,0,0], label='up')
line5 = ax.plot(corr_down[:,0,0], label='down')
ax.legend()
ax.set_ylim([0, 3.2])

tracker1 = Tracker2D([corr_const1, corr_const2, corr_const3, corr_up, corr_down], fig, ax, [line1, line2, line3, line4, line5])
fig.canvas.mpl_connect('scroll_event', tracker1.on_scroll)


fig, ax = plt.subplots()
line1 = ax.plot(fixed_const1[:,0,0], label='const1')
line2 = ax.plot(fixed_const2[:,0,0], label='const2')
line3 = ax.plot(fixed_const3[:,0,0], label='const3')
line4 = ax.plot(fixed_up[:,0,0], label='up')
line5 = ax.plot(fixed_down[:,0,0], label='down')
ax.legend()
ax.set_ylim([0, 3.2])

tracker2 = Tracker2D([fixed_const1, fixed_const2, fixed_const3, fixed_up, fixed_down], fig, ax, [line1, line2, line3, line4, line5])
fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)


plt.show()
