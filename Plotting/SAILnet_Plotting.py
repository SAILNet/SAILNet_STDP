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

with open('output_no_stdp.pkl','rb') as f:
    W,Q,theta,stdp,magnitude_stdp,magnitude_dW = cPickle.load(f)

# <codecell>




def Plot_RF(Q):
    im_size, num_dict = Q.shape

    side = int(np.round(np.sqrt(im_size)))
    OC = num_dict/im_size


    img = tile_raster_images(Q.T, img_shape = (side,side), tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
    plt.imshow(img,cmap=plt.cm.Greys)
    plt.title('Receptive Fields')
    plt.imsave('RF_no_stdp.png', img, cmap=plt.cm.Greys)
    plt.show()

    


def PlotdWstdp(stdp,dW):
    plt.plot(stdp, color="green", label="STDP")
    plt.plot(dW,color="blue", label="dW")
    plt.legend(bbox_to_anchor=(1,.5))
    plt.savefig('stdp_dW.png')



Plot_RF(Q)

plt.close
plt.clf


PlotdWstdp(magnitude_stdp,magnitude_dW)




"""
with open('output_stdp.pkl','rb') as f:
    W_stdp,Q,theta,stdp,magnitude_stdp,magnitude_dW = cPickle.load(f)

# <codecell>

im_size, num_dict = Q.shape

# <codecell>

side = int(np.round(np.sqrt(im_size)))
OC = num_dict/im_size

# <codecell>

img = tile_raster_images(Q.T, img_shape = (side,side), tile_shape = (2*side,side*OC/2), tile_spacing=(1, 1), scale_rows_to_unit_interval=True, output_pixel_vals=True)
plt.imshow(img,cmap=plt.cm.Greys)
plt.title('Receptive Fields')
plt.imsave('RF_stdp.png', img, cmap=plt.cm.Greys)
plt.show()


#


# <codecell>

plt.imshow(W,cmap=plt.cm.Greys)
plt.title('Inhibitory Connections')
plt.show()

plt.close

print np.std(W)
print np.std(W_stdp)

plt.hist2d(W,W_stdp)
"""

# <codecell>


