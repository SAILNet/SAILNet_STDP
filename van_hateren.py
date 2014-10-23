import numpy as np
import os
import array

# Olshausen 2013 subset of van Hateren dataset
olshausen_subset = ['imk00264.iml',
                    'imk00315.iml',
                    'imk00665.iml',
                    'imk00695.iml',
                    'imk00735.iml',
                    'imk00765.iml',
                    'imk00777.iml',
                    'imk00944.iml',
                    'imk00968.iml',
                    'imk01026.iml',
                    'imk01042.iml',
                    'imk01098.iml',
                    'imk01251.iml',
                    'imk01306.iml',
                    'imk01342.iml',
                    'imk01726.iml',
                    'imk01781.iml',
                    'imk02226.iml',
                    'imk02260.iml',
                    'imk02262.iml',
                    'imk02982.iml',
                    'imk02996.iml',
                    'imk03332.iml',
                    'imk03362.iml',
                    'imk03401.iml',
                    'imk03451.iml',
                    'imk03590.iml',
                    'imk03686.iml',
                    'imk03751.iml',
                    'imk03836.iml',
                    'imk03848.iml',
                    'imk04099.iml',
                    'imk04103.iml',
                    'imk04172.iml',
                    'imk04207.iml']

class VanHateren:
    def __init__(self, image_dir, use_olshausen=False):
        if use_olshausen:
            self.im_files = [os.path.join(image_dir, ols_file) for ols_file in olshausen_subset]
        else:
            import glob
            self.im_files = glob.glob(os.path.join(image_dir, '*.iml'))
        print len(self.im_files)
        print image_dir

    def load_images(self, n_images, rng=None):
        if rng is None:
            im_indicies = np.arange(n_images)
        else:
            im_indicies = rng.permutation(len(self.im_files))[:n_images]

        imgset = np.zeros((n_images, 1024, 1024), dtype='float32')
        for i,index in enumerate(im_indicies):
            with open(self.im_files[index], 'rb') as handle:
                s = handle.read()
            arr = array.array('H', s)
            arr.byteswap()
            img = np.array(arr, dtype='uint16').reshape(1024, 1536)

            # Olshausen 2013 extracts central 1024x1024 patch
            img = img[:, (1536-1024)/2:(1536+1024)/2]
            img = np.log(img.astype('float32')+0.1)
            # Remove image mean
            img -= img.mean()
            img = img/(img.max()*1.0)
            imgset[i] = img

        return imgset
