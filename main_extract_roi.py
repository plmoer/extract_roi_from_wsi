import numpy as np
import os
import time
import openslide
from PIL import Image
from utils.misc import *
from skimage.segmentation import slic
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries


# in_dir = '/media/linmin/My Passport/myResearch/data_sets/CPM-RadPath_2019_Training_Data'
in_dir = '/home/linmin/myResearch/deep_learning/CPM19_tumor_code/train'
wsi_dir = os.path.join(in_dir, 'temp')
# wsi_dir = os.path.join(in_dir, 'Pathology')
out_dir = 'pathology_ori'
sub_dir = 'region_selection_location'
if os.path.exists(out_dir) == False:
    os.mkdir(out_dir)

if os.path.exists(sub_dir) == False:
    os.mkdir(sub_dir)

lv = 9
percentile = 10  # percentile to select region
tile_size = 2048  # tile size of selected roi of each wsi


def main():
    all_files = os.listdir(wsi_dir)
    n_file = len(all_files)
    for idx, f in enumerate(all_files):
        bValid = False
        nLoop = 0  # control the loop times
        print('\n....working on %s: %d/%d' % (f, idx+1, n_file))
        wsi_file_path = os.path.join(wsi_dir, f)  # get full path
        thumbnail = get_thumbnail(wsi_file_path, lv)  # get thumbnail at the lv
        # thumbnail = np.array(Image.open('map_lv6.bmp'))
        segments = slic(thumbnail, 100, sigma=5)
        temp_segments = segments.copy()
        ave_list, valid_Ele = get_valid_area(thumbnail, percentile, segments)

        # plt.imshow(mark_boundaries(thumbnail, segments))
        # plt.show()
        nMaxLoop = len(valid_Ele)-1

        while np.logical_and(bValid == False, nLoop < nMaxLoop):  # try different regions
            # get region at the given percentile
            temp_med_idx = get_region_idx(ave_list, percentile)
            value_in_segs = valid_Ele[temp_med_idx]
            ratio_x, ratio_y = get_location(  # get relative position in ratio
                temp_med_idx, valid_Ele, percentile, segments)

            # try different center points within the region
            while np.logical_and(bValid == False, nLoop < 20):
                nFactor = np.random.uniform(0.7, 1.3)
                ratio_x_temp, ratio_y_temp = ratio_x*nFactor, ratio_y*nFactor
                roi = get_roi(wsi_file_path, tile_size,
                              ratio_x_temp, ratio_y_temp)
                bValid = verification(roi)
                nLoop = nLoop + 1
            invalid_index = ave_list.index(max(ave_list))
            # remove the region with larest mean intensity
            ave_list.pop(invalid_index)
            valid_Ele.pop(invalid_index)

            temp_segments[np.where(temp_segments == value_in_segs)] = 600*nLoop
            print('.. loop times:', nLoop)

        # plt.imshow(mark_boundaries(thumbnail, segments))
        # plt.imshow(temp_segments, alpha=0.5)
        # plt.show()

        boundary = mark_boundaries(thumbnail, segments)

        boundary = boundary*255

        bound_obj = Image.fromarray(boundary.astype(np.uint8))

        nMax = np.max(temp_segments)
        temp_segments = temp_segments/nMax*255

        temp_segments_obj = Image.fromarray(
            temp_segments.astype(np.uint8), 'L')
        new_img = mergeImage(temp_segments_obj, bound_obj, 0.5)
        new_img = new_img.convert("RGB")
        new_img.save(os.path.join(sub_dir, f[:-5]+'.jpg'))

        roi_rgb = roi.convert('RGB')
        roi_rgb.save(os.path.join(out_dir, f[:-5]+'.jpg'))


if __name__ == '__main__':
    startTime = time.time()
    main()
    endTime = time.time()
    print('****** It takes %d seconds to complete***' % (endTime-startTime))
