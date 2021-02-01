from time import time
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ir_sim import dpd

im = Image.open('test-image03cross.png')
a = np.array(im)

#r, g, b, _ = a.swapaxes(0,1).swapaxes(0,2)
#gray = 0.2998*r + 0.5870*g + 0.1140*b
gray = a

ksize = 15


plt.figure()
plt.imshow(gray, cmap='gray')


#######
# Chan
#######


start_time = time()
deadc = dpd.dpd_chan(gray)
print("Chan %s seconds" % (time() - start_time))
fixedc = dpd.replace_dead_conv(gray, deadc, ksize=ksize)

plt.figure()
plt.imshow(deadc, vmin=0, vmax=1, cmap='Reds')
plt.title('chan')

plt.figure()
plt.imshow(fixedc, cmap='gray')
plt.title('chan')

###############
# Chao et. al.
###############

# v1
start_time = time()
dead = dpd.dpd(gray)
print("Chao v1 %s seconds" % (time() - start_time))
fixed = dpd.replace_dead(gray, dead)

plt.figure()
plt.imshow(dead, vmin=0, vmax=1, cmap='Reds')
plt.title('chao')

plt.figure()
plt.imshow(fixed, cmap='gray')
plt.title('chao')

# v2
start_time = time()
deadt = dpd.testdpd(gray, L=5, d=1)
print("Chao v2 %s seconds" % (time() - start_time))
fixedt = dpd.replace_dead_conv(gray, deadt, ksize=ksize)

plt.figure()
plt.imshow(deadt, vmin=0, vmax=1, cmap='Reds')
plt.title('chao2')

plt.figure()
plt.imshow(fixedt, cmap='gray')
plt.title('chao2')


################
# Takam et. al.
################

start_time = time()
deadtk = dpd.dpd_takam(gray)
print("Takam %s seconds" % (time() - start_time))
fixedtk = dpd.replace_dead_conv(gray, deadtk, ksize=ksize)

plt.figure()
plt.imshow(deadtk, vmin=0, vmax=1, cmap='Reds')
plt.title('takam')

plt.figure()
plt.imshow(fixedtk, cmap='gray')
plt.title('takam')

plt.show()
