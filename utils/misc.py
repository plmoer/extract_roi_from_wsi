"""
Date: Aug. 15, 2019
Author: Linmin
"""
import os
import numpy as np
import openslide
from PIL import Image
import time
import random
import matplotlib.pyplot as plt
from scipy.ndimage.measurements import center_of_mass

"""
the function is to merge two images
input:  img1-->image 1 in Image, not array
        img2-->image 2 in Image, not array
return: imge-->merged image, not array
"""


def mergeImage(img1, img2, alpha=0.5):
    assert img1.size == img2.size
    # [x, y, z] = img1.shape
    new_img_1 = Image.new('RGBA', img1.size, color=(0, 0, 0, 0))
    new_img_1.paste(img1, (0, 0))
    new_img_1.paste(img2, (0, 0))

    new_img_2 = Image.new('RGBA', img1.size, color=(0, 0, 0, 0))
    new_img_2.paste(img2, (0, 0))
    new_img_2.paste(img1, (0, 0))

    img = Image.blend(new_img_1, new_img_2, alpha)

    return img


"""
the function is to verify the roi 
input:  roi-->select region in Image, not array
        ration_x, ration_y -->relative position
return: bValid-->confirmation
        new_ratio_x, new_ratio_y
"""


def verification(roi):
    roi_gray = np.array(roi.convert('L'))
    # m, n = roi_gray.size

    ave = np.median(roi_gray)  # median
    nMin = np.min(roi_gray)
    # number of large intensity
    # n_large = np.where(np.array(roi_gray) > 200)[0].size
    # if np.logical_and(ave < 200, n_large < m*n*0.3):
    if np.logical_and(ave < 200, nMin > 10):
        bValid = True
    else:
        bValid = False
        # factor_x = np.random.uniform(low=0.7, high=1.3, size=1)
        # factor_y = np.random.uniform(low=0.7, high=1.3, size=1)
        # ratio_x, ratio_y = ratio_x*factor_x, ratio_y*factor_y

    return bValid


"""
the function is to get the tile as size at relative x,y
input:  wsi-file_path-->full path of the wsi
        tile_size-->tile size
        ratio_x, ratio_y-->center at relative position
return: selected roi in PIL
"""


def get_roi(wsi_file_path, tile_size, ratio_x, ratio_y):
    slidePtr = openslide.OpenSlide(wsi_file_path)
    [n, m] = slidePtr.dimensions  # dimension
    y, x = np.round(n*ratio_y).astype(np.uint16), np.round(m *
                                                           ratio_x).astype(np.uint16)  # get center location
    roi = slidePtr.read_region(
        (y, x), 0, (tile_size, tile_size))  # get the roi
    return roi


"""the function is to get the ratio position by taking the median intensity area thumbnail and segments are image array
input: thumbnail-->thumbnail image in array
        percentile-->percentile to choose the average intensity at this percentile
        segments-->oversegmented image in array
return: ratio_x, ratio_y-->ratio position at
"""


def get_valid_area(thumbnail, percentile, segments):
    # ratio_x, ratio_y = 0, 0
    thumbnail_obj = Image.fromarray(thumbnail).convert('L')  # to gray
    thumbnail_gray = np.array(thumbnail_obj)  # to array
    nElement = np.unique(segments).shape[0]
    ave_list = []
    valid_Ele = []
    for i in range(nElement):
        area = thumbnail_gray[np.where(segments == i)]  # for each region
        # area = temp_area[np.nonzero(temp_area)]  # remove 0
        ave = np.mean(area)  # compute mean intensity
        # ave = np.std(area)  # compute mean intensity
        # if np.logical_and(ave < 30, ave > 75):  # discard region having too much white
        #     pass
        # else:
        #     ave_list.append(ave)
        #     valid_Ele.append(i)
        ave_list.append(ave)
        valid_Ele.append(i)

    return ave_list, valid_Ele


def get_region_idx(ave_list, percentile):

    temp_med_idx = ave_list.index(np.percentile(
        ave_list, percentile, interpolation='nearest'))
    return temp_med_idx


def get_location(temp_med_idx, valid_Ele, percentile, segments):

    med_Ele = valid_Ele[temp_med_idx]
    # x, y = np.where(segments == med_Ele)
    # center_x, center_y = np.median(x), np.median(y)
    new_segments = segments.copy()
    new_segments = (new_segments == med_Ele).astype(int)
    [center_x, center_y] = center_of_mass(new_segments)
    ratio_x, ratio_y = center_x/segments.shape[0], center_y/segments.shape[1]

    return ratio_x, ratio_y


"""
The function is to get thumbnail image at the lv of wsi
input:  wsi_file_path-->full path to the wsi
        lv--> the level
return: thumbnail in array
"""


def get_thumbnail(wsi_file_path, lv):
    slidePtr = openslide.OpenSlide(wsi_file_path)
    Lcnt = slidePtr.level_count
    print('number of layers =', Lcnt)
    [n, m] = slidePtr.dimensions
    # print('dimension =', n, 'x', m)
    if min(slidePtr.dimensions) < 50000:
        lv = lv // 2

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
    # print('mpp_x =', mpp_x)
    # print('vendor name :', prop.get('openslide.vendor'))
    # print('magnification =', prop.get('openslide.objective-power'))

    tic = time.time()
    # lv = 10
    if(Lcnt == 1):
        ratio = 2**lv
        gap = 2**(lv-1)
        x_lv = n//ratio
        y_lv = m//ratio
        # print('level size =', x_lv, 'x', y_lv)
        map_lv = np.zeros((y_lv, x_lv, 3), dtype=np.uint8)
        for i in range(1, y_lv-1):
            for j in range(1, x_lv-1):
                x = j*ratio - gap
                y = i*ratio - gap
                ARGB = np.array(slidePtr.read_region((x, y), 0, (1, 1)))
                map_lv[i, j, 0] = ARGB[0, 0, 2]
                map_lv[i, j, 1] = ARGB[0, 0, 1]
                map_lv[i, j, 2] = ARGB[0, 0, 0]

        # map_name = 'map_lv'+str(lv)+'.bmp'
        # cv2.imwrite(map_name, map_lv)
    toc = time.time()
    print('Level =', lv, 'Elapsed time =', toc-tic, 'sec')
    return map_lv
