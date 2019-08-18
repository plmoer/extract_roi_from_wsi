'''
For BraTS 2018 training and validation.
after normalization image within 0 and 1, the mean of flair, t1, t1ce, t2 is [0.4484 0.4515 0.4502 0.4337]
the std of flair, t1, t1ce, t2 is [0.1923 0.1633 0.1675 0.2183]

'''
import numpy as np
import torch
import time
import os
import pandas as pd
from torch.autograd import Variable
import zipfile
from glob import glob
# from utils.getImageData import *
import nibabel as nib
# from utils.jpu_net import JPUNet


rootDir = '/home/linmin/Desktop/CPM19_tumor/input-images'
in_dir = os.path.join(rootDir, 'Radiology')
unzip_dir = os.path.join(rootDir, 'Radiology_unzip')
uncertainty_dir = os.path.join(rootDir, 'Radiology_uncertainty')
model_uncertainty = 'model_last_vae_309'


def loadModel(model_uncertainty):
    model = torch.load(model_uncertainty)
    return model


def get_pat_list(in_dir):
    temp_patList = glob(in_dir+'/*')
    patList = [w.replace(in_dir+'/', '')
               for w in temp_patList]  # get zip file name
    patList = [w.replace('.zip', '') for w in patList]  # only keep patient id
    return patList


def uncompress(in_dir, unzip_dir):
    temp_patList = glob(in_dir+'/*')
    for idx, pat in enumerate(temp_patList):
        print('...unzip file: %s, %d/%d' % (pat, idx+1, len(temp_patList)))
        with zipfile.ZipFile(pat) as zip_ref:
            zip_ref.extractall(unzip_dir)


def exec_uncertainty(patList, unzip_dir, uncertainty_dir, uncertainty_model):
    if os.path.exists(uncertainty_dir) == False:  # create the fold
        os.mkdir(uncertainty_dir)

    for idx, pat in enumerate(patList):
        print('......uncertainty maps: %s, %d/%d' % (pat, idx+1, len(patList)))

    return 0


def main(in_dir):
    patList = get_pat_list(in_dir)  # get patient id list
    # uncompress(in_dir, unzip_dir)  # unzip all files
    uncertainty_model = loadModel(model_uncertainty)
    exec_uncertainty(patList, unzip_dir, uncertainty_dir, uncertainty_model)

    pass


if __name__ == '__main__':
    startTime = time.time()
    main(in_dir)
    endTime = time.time()
    print('*** It takes %d seconds to complete***' % (endTime-startTime))
