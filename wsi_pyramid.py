'''

Topic: Generate WSI pyramidal structure

Create Date: 2019/08/15

Modified Date: 2019/08/15

Author: David Hsu

'''

import numpy as np
import openslide
import time
import cv2
import os

file_dir = os.getcwd()+'/Pathology/'
all_files = os.listdir(file_dir)
n_file = len(all_files)

for f in range(n_file):
    wsi_file_path = file_dir + all_files[f]
    print(all_files[f])
    slidePtr = openslide.OpenSlide(wsi_file_path)
    Lcnt = slidePtr.level_count
    print('number of layers =', Lcnt)
    [n, m] = slidePtr.dimensions
    print('dimension =', n, 'x', m)

    '''
    size_in_each_level = np.zeros((Lcnt,2),dtype=np.int)
    for i in range(Lcnt):
        size_in_each_level[i,:] = slidePtr.level_dimensions[i] 
    '''

    prop = slidePtr.properties
    mpp_x = prop.get('openslide.mpp-x')
    if(mpp_x == None):
        XResolution = float(prop.get('tiff.XResolution'))
        mpp_x = 10**4/XResolution
    print('mpp_x =', mpp_x)
    print('vendor name :', prop.get('openslide.vendor'))
    print('magnification =', prop.get('openslide.objective-power'))

    tic = time.time()
    lv = 10
    if(Lcnt == 1):
        ratio = 2**lv
        gap = 2**(lv-1)
        x_lv = n//ratio
        y_lv = m//ratio
        print('level size =', x_lv, 'x', y_lv)
        map_lv = np.zeros((y_lv, x_lv, 3), dtype=np.uint8)
        for i in range(1, y_lv-1):
            for j in range(1, x_lv-1):
                x = j*ratio - gap
                y = i*ratio - gap
                ARGB = np.array(slidePtr.read_region((x, y), 0, (1, 1)))
                map_lv[i, j, 0] = ARGB[0, 0, 2]
                map_lv[i, j, 1] = ARGB[0, 0, 1]
                map_lv[i, j, 2] = ARGB[0, 0, 0]

        map_name = 'map_lv'+str(lv)+'.bmp'
        cv2.imwrite(map_name, map_lv)
    toc = time.time()
    print('\nLevel =', lv, 'Elapsed time =', toc-tic, 'sec')

    print('')
