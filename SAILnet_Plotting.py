# -*- coding: utf-8 -*-
# <nbformat>3.0</nbformat>

# <headingcell level=1>

# SAILnet Testing

# <codecell>

import cPickle
import matplotlib.pyplot as plt
import numpy as np
from utils import tile_raster_images
#%matplotlib inline 

# <codecell>

with open('output.pkl','rb') as f:
    W,Q,theta = cPickle.load(f)

# <codecell>

im_size, num_dict = Q.shape

# <codecell>

side = int(np.round(np.sqrt(im_size)))
OC = num_dict/im_size

# <codecell>

img = tile_raster_images(Q.T, img_shape = (side,side), tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys)
plt.title('Receptive Fields')
plt.show()

# <codecell>

plt.imshow(W,cmap=plt.cm.Greys)
plt.title('Inhibitory Connections')
plt.show()

# <codecell>


