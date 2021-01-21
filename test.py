import os

import numpy as np
import matplotlib.pyplot as plt

from ir_sim import sys_param as sp
from ir_sim import tolerance as t
from ir_sim import blackbody as bb
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

###############
# Parameters
###############

R_tol = g_tol = c_tol = 1e-5
T = np.linspace(300, 400, 10)
pix_v = 100
pix_h = 120
pix_v_all = pix_v + 2*(sp.boundary_pix + sp.skimming_pix)
pix_h_all = pix_h + 2*(sp.boundary_pix + sp.skimming_pix)

# C may be 0.8 (lowest gain), 0.4, 0.2 and 0.1 pF (highest gain)
C = 0.4e-12

gen_params = {
    'sensor': (pix_v_all, pix_h_all, sp.skimming_pix, sp.boundary_pix),
    'opamp': (sp.R1, sp.R2, sp.R3, C)
}


try:
    frames = np.loadtxt('data_test1/frames.txt').reshape(T.size, pix_v, pix_h)
    tol = t.load_data('data_test1')
    P = np.loadtxt('data_test1/P.txt').reshape(T.size, pix_v, pix_h)
except:
    Pcam = bb.integration(T)
    P = bb.power_distribution_over_sens_area(Pcam, size=(pix_v, pix_h))
    
    tol = t.generate_arrays(R_tol, g_tol, c_tol, size=(pix_v_all, pix_h_all))
    gen_params['tol'] = tol
    
    gg = g.FrameGen(**gen_params)
    frames_all = gg.run_frames(P, Pcam[0])
    frames = gg.filter_actives(frames_all)
    
    
    try:
        os.mkdir('data_test1')
    except:
        pass
    
    t.save_data('data_test1', *tol)
    
    with open('data_test1/frames.txt', 'w') as outf:
        for f in frames:
            np.savetxt(outf, f)
    
    with open('data_test1/P.txt', 'w') as outf:
        for p in P:
            np.savetxt(outf, p)

###############
# Correction
###############

dp = dpd.dpd(frames)
satp = dpd.saturated_pixels(frames)
stuck = dpd.stuck_pixels(frames)
dp = np.logical_or(dp, stuck)

frames_fixed = dpd.replace_dead(frames, dp)

coefs = nuc.coeff_calc(frames_fixed, T, points=3, quad=1)
corr = nuc.nuc(frames_fixed, coefs)

dp2 = dpd.dpd(corr, lim=0.01)
corr_fixed = dpd.replace_dead(corr, dp2)

coefs_lt = nuc.coeff_linear_temp(corr_fixed, T)
corr_lt = nuc.linear_temp(corr_fixed, coefs_lt)


##########
# Figures
##########

plt.figure()
plt.title('power distribution')
plt.imshow(P[-1])

_, _, t = imshow_i(frames, title='active pixels', vmin=0, vmax=3.2)

plt.figure()
plt.imshow(dp, vmin=0, vmax=1, cmap='Reds')
plt.title('dead pixels')

plt.figure()
plt.imshow(satp, vmin=0, vmax=1, cmap='Reds')
plt.title('saturated pixels')

plt.figure()
plt.imshow(frames_fixed[-1], vmin=0, vmax=3.2)
plt.title('active pixels !dead')

'''
plt.figure()
plt.imshow(coefs[0])
plt.title('b')

plt.figure()
plt.imshow(coefs[1])
plt.title('k')
'''

_, _, t2 = imshow_i(corr, title='nuc', vmin=0, vmax=3.2)

plt.figure()
plt.imshow(dp2, vmin=0, vmax=1, cmap='Reds')
plt.title('dead pixels 2')

_, _, t3 = imshow_i(corr_fixed, title='nuc !dead', vmin=0, vmax=3.2)


fig, ax = plt.subplots()
line1 = ax.plot(frames[:,0,0], label='raw')
line2 = ax.plot(corr[:,0,0], label='corrected')
ax.legend()
ax.set_ylim([0, 3.2])

tracker1 = Tracker2D([frames, corr_fixed], fig, ax, [line1, line2])
fig.canvas.mpl_connect('scroll_event', tracker1.on_scroll)


plt.figure()
v_avg = np.average(corr_fixed, (1,2))
plt.plot(T, v_avg)
plt.xlabel('T')
plt.ylabel('V')

plt.figure()
v = np.linspace(0, 3.2, 1000)
Tpol = coefs_lt[0]*v**2+coefs_lt[1]*v+coefs_lt[2]
plt.plot(v, Tpol)
plt.xlabel('V')
plt.ylabel('T')

fig, ax = plt.subplots()
line1 = ax.plot(T, corr_lt[:,0,0])

tracker2 = Tracker2D([corr_lt], fig, ax, [line1])
fig.canvas.mpl_connect('scroll_event', tracker2.on_scroll)


plt.show()
