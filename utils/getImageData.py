import numpy as np
import nibabel as nib
from glob import glob


def getImageData(imgDir, patientName, bGT=0):
    if bGT == 1:
        # list keyword to find T1, T1C, T2, FLAIR, and Ground Truth images
        keyword = ['_flair.nii.gz', '_t1.nii.gz',
                   '_t1ce.nii.gz', '_t2.nii.gz', '_seg.nii.gz']
    elif bGT == 0:
        keyword = ['_flair.nii.gz', '_t1.nii.gz', '_t1ce.nii.gz',
                   '_t2.nii.gz']  # list keyword to find T1, T1C, T2, FLAIR,

    imgData = np.array([])
    for idx, modality in enumerate(keyword):
        # get full path of each modality. 0 is used to get the first string
        mod_path = glob(imgDir+'/' + patientName+'/*' + modality)[0]
        img = nib.load(mod_path)  # get nib object
        img = img.get_data()  # convert to numpy array
        if imgData.size == 0:
            # create a big zero array. nChannel*nRow*nColoum*nSlice
            imgData = np.zeros(
                (len(keyword), img.shape[0], img.shape[1], img.shape[2]))
        imgData[idx] = img  # save all modalities into one big data array
    return imgData
