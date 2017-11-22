import numpy as np
import gaussian_filter2D
from gaussian_filter2D import gauss_kern2D
from gaussian_filter2D import gauss_filt2D
import matplotlib
from matplotlib import pyplot as plt

imgs = np.memmap('/home/zlazri/Documents/spontdata/Image_0001_0001.raw', dtype = 'uint16', mode = 'r', shape = (15000, 4, 512, 512), order ='C')

img1 = imgs[1,1,:,:]

Gkernel = gauss_kern2D(3, 1)

Gimg = gauss_filt2D(img1, Gkernel)

plt.imshow(Gimg)
plt.show()
