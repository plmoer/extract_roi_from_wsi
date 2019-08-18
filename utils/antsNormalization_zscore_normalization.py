"""
The function is to normalize MRI with n4 Bias correction. Apply z-score normalization
and force the intensity within [0, 1]
Revised by Linmin
"""
import os
from glob import glob
import nibabel as nib
import numpy as np
import subprocess
import time

# np.random.seed(5)  # for reproducibility
# progress = progressbar.ProgressBar(widgets=[progressbar.Bar('*', '[', ']'), progressbar.Percentage(), ' '])


class BrainPipeline(object):
    '''
    A class for processing brain scans for one patient
    INPUT:  (1) filepath 'path': path to directory of one patient. Contains following mha files:
            flair, t1, t1c, t2, ground truth (gt)
            (2) bool 'n4itk': True to use n4itk normed t1 scans (defaults to True)
            (3) bool 'n4itk_apply': True to apply and save n4itk filter to t1 and t1c scans for given patient. This will only work if the
    '''

    def __init__(self, path, n4itk=False, n4itk_apply=True, bGT=0):
        self.path = path
        self.n4itk = n4itk
        self.n4itk_apply = n4itk_apply
        self.bGT = bGT
        # temprary name for bias correction result
        self.tempN4itk_name = '_tempN4itk.nii.gz'
        self.deleteBiasCorrectionFile = True  # whether delete the bias correction file
        if bGT == 0:
            self.modes = ['flair.nii.gz', 't1.nii.gz',
                          't1ce.nii.gz', 't2.nii.gz']
        elif bGT == 1:
            self.modes = ['flair.nii.gz', 't1.nii.gz',
                          't1ce.nii.gz', 't2.nii.gz', 'seg.nii.gz']

        # slices=[[flair x 155], [t1], [t1c], [t2], [gt]], 155 per modality
        self.imgData = self.read_scans()
        self.normed_Data = self.norm_slices()

    def read_scans(self):
        '''
        goes into each modality in patient directory and loads individual scans.
        transforms scans of same slice into strip of 5 images
        '''
        imgData = np.array([])
        modes = self.modes
        # imgData = np.zeros((len(modes), 155, 240, 240))
        for idx, modality in enumerate(modes):
            # get full path of each modality. 0 is used to get the first string
            mod_path = glob(self.path+'/*'+modality)[0]
            print('......Loading %s ' % (mod_path))
            img = nib.load(mod_path)  # get nib object
            img = img.get_data()  # convert to numpy array
            if imgData.size == 0:
                # create a big zero array. nChannel*nRow*nColoum*nSlice
                imgData = np.zeros(
                    (len(modes), img.shape[0], img.shape[1], img.shape[2]), dtype=np.float16)
            if self.n4itk_apply == True:  # if need bias correction
                if modality.find("t1") != -1:  # only applys to t1 and t1ce modality
                    img = np.array([])  # empty the img
                    self.n4itk_norm(mod_path)  # apply bias correction
                    # get the bias corrected file
                    temp_mod = glob(
                        self.path+'/*'+modality[:-7]+self.tempN4itk_name)[0]
                    img = nib.load(temp_mod)  # get nib object
                    img = img.get_data()  # convert to numpy array
                    if self.deleteBiasCorrectionFile == True:  # delete the bias correction file
                        os.remove(temp_mod)
            imgData[idx] = img  # save all modalities into one big data array
        return imgData

    def norm_slices(self):
        '''
        normalizes each slice in self.slices_by_slice, excluding gt
        subtracts mean and div by std dev for each slice
        clips top and bottom one percent of pixel intensities
        if n4itk == True, will apply n4itk bias correction to T1 and T1c images
        '''
        print('Normalizing slices...')
        imgData = self.imgData  # nChannel*nRow*nColumn*nSlice
        # nChannel*nSlice*nRow*nColumn
        imgData = np.transpose(imgData, (0, 3, 1, 2))
        normed_data = np.zeros(imgData.shape, dtype=np.float64)
        for idx in range(len(imgData)):
            if idx < 4:  # exclusive ground truth if exists. 0-flair, 1-t1, 2-t1ce, 3-t2
                img = imgData[idx]
                # b, t = np.percentile(img, (0.5, 99.5))
                # img = np.clip(img, b, t)
                bg_index = np.where(img == 0)
                fg_index = np.where(img != 0)
                n_mean = np.mean(img[fg_index])
                n_std = np.std(img[fg_index], dtype=np.float64)
                img = (img-n_mean)/n_std
                # temp_min = np.min(img)
                # temp_max = np.max(img)
                # img = (img-temp_min)/(temp_max-temp_min)
                # img = np.clip(img, -5, 5)  # clip image
                # img = img/10+0.5  # force to 0 and 1
                img[bg_index] = 0  # set background as 0
                normed_data[idx] = img
            else:
                # if ground truth exists, directly store the ground truth without any normalization
                normed_data[idx] = imgData[idx]
        return normed_data  # nChannel*nSlice*nRow*nColumn

    def save_zscore(self, reg_norm_n4, patient_num, outDir):
        pid = (self.path).split('/')[-1]  # get patient ID
        pidDir = os.path.join(outDir, pid)
        modes = self.modes
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        if not os.path.exists(pidDir):
            os.mkdir(pidDir)

        if len(modes) == 5:
            normed_data = self.normed_Data[0:4]  # get flair, t1, t1ce, and t2
            # get ground truth #nSlice*nRow*nColumn
            gt_data = self.normed_Data[-1]
            # ground truth with nRow*nColumn*nSlice
            gt_data = np.transpose(gt_data, (1, 2, 0))
            gt_data = gt_data.astype(np.uint8)
            gt = nib.Nifti1Image(gt_data, None)
            savedFullName = os.path.join(
                outDir, pid + '/' + pid + '_' + modes[-1])
            nib.save(gt, savedFullName)
        else:
            normed_data = self.normed_Data  # nChannel*nSlice*nRow*nColumn

        for idx in range(len(normed_data)):
            img = normed_data[idx]
            img = np.transpose(img, (1, 2, 0))
            img = nib.Nifti1Image(img, None)
            savedFullName = os.path.join(
                outDir, pid + '/' + pid + '_' + modes[idx])
            nib.save(img, savedFullName)

    def save_patient(self, reg_norm_n4, patient_num, outDir):
        '''
        INPUT:  (1) int 'patient_num': unique identifier for each patient
                (2) string 'reg_norm_n4': 'reg' for original images, 'norm' normalized images, 'n4' for n4 normalized images
        OUTPUT: saves png in Norm_PNG directory for normed, Training_PNG for reg
        '''
        # print ('Saving scans for patient {}...'.format(patient_num))
        pid = (self.path).split('/')[-1]  # get patient ID
        pidDir = os.path.join(outDir, pid)
        modes = self.modes
        if not os.path.exists(outDir):
            os.mkdir(outDir)
        if not os.path.exists(pidDir):
            os.mkdir(pidDir)

        if len(modes) == 5:
            normed_data = self.normed_Data[0:4]  # get flair, t1, t1ce, and t2
            # get ground truth #nSlice*nRow*nColumn
            gt_data = self.normed_Data[-1]
            # ground truth with nRow*nColumn*nSlice
            gt_data = np.transpose(gt_data, (1, 2, 0))
            gt_data = gt_data.astype(np.uint8)
            gt = nib.Nifti1Image(gt_data, None)
            savedFullName = os.path.join(
                outDir, pid + '/' + pid + '_' + modes[-1])
            nib.save(gt, savedFullName)
        else:
            normed_data = self.normed_Data  # nChannel*nSlice*nRow*nColumn
        # nSlice*nChannel*nRow*nColumn
        normed_data = np.transpose(normed_data, (1, 0, 2, 3))
        new_data = np.zeros(normed_data.shape)
        for slice_ix in range(len(normed_data)):  # reshape to strip
            # strip = normed_data[slice_ix].reshape(960, 240)
            strip = normed_data[slice_ix]
            if np.max(strip) != 0:  # set values < 1
                strip /= np.max(strip)
            if np.min(strip) <= -1:  # set values > -1
                strip /= abs(np.min(strip))
            new_data[slice_ix] = strip
        # nchannel*nRow*nColum*nSlice
        new_data = np.transpose(new_data, (1, 2, 3, 0))

        for idx in range(len(new_data)):
            img = new_data[idx]
            img = np.clip(img, 0, 1)
            img = img * 255
            img = img.astype(np.uint8)
            img = nib.Nifti1Image(img, None)
            if os.path.exists(os.path.join(outDir, pid)) == False:
                os.mkdir(os.path.join(outDir, pid))
            savedFullName = os.path.join(
                outDir, pid + '/' + pid + '_' + modes[idx])
            nib.save(img, savedFullName)

        # maxValue = np.max(normed_data)
        # minValue = np.min(normed_data)
        # for idx in range(len(normed_data)):
        #     strip = normed_data[idx]
        #     strip = (strip-minValue)/(maxValue-minValue)*255
        #     strip = strip.astype(np.uint8)
        #     strip = nib.Nifti1Image(strip, None)
        #     savedFullName = os.path.join(outDir, pid+'/'+pid+'_'+modes[idx])
        #     nib.save(strip, savedFullName)

        # temp_data = np.transpose(normed_data, (3,0,1,2))
        # new_data = np.zeros(temp_data.shape)
        # for idx in range(len(temp_data)):
        #     strip = temp_data[idx]
        #     if np.max(strip) != 0:  # set values < 1
        #         strip /= np.max(strip)
        #     if np.min(strip) <= -1:  # set values > -1
        #         strip /= abs(np.min(strip))
        #     new_data[idx] = strip
        # new_data = np.transpose(new_data, (0,1,2,3))
        # for idx in range(len(new_data)):
        #     strip = normed_data[idx]
        #     strip = (((strip- minValue) * (1 - 0)) / (maxValue - minValue))
        #     # strip = (strip-minValue)/(maxValue-minValue)*255
        #     strip = strip*255
        #     strip = strip.astype(np.uint8)
        #     strip = nib.Nifti1Image(strip, None)
        #     savedFullName = os.path.join(outDir, pid+'/'+pid+'_'+modes[idx])
        #     nib.save(strip, savedFullName)

    def n4itk_norm(self, path, n_dims=3, n_iters='[20,20,10,5]'):
        '''
        INPUT:  (1) filepath 'path': path to mha T1 or T1c file
                (2) directory 'parent_dir': parent directory to mha file
        OUTPUT: writes n4itk normalized image to parent_dir under orig_filename_n.mha
        '''
        # print('Patient: {}'.format(path))

        output_fn = path[:-7] + self.tempN4itk_name
        # print '{}'.format(path)
        # run n4_bias_correction.py path n_dim n_iters output_fn
        subprocess.call('python n4_bias_correction.py ' + path + ' ' + str(n_dims) + ' ' + n_iters + ' ' + output_fn,
                        shell=True)


def save_patient_slices(patients, type, bGT, imgDir, outDir):
    '''
    INPUT   (1) list 'patients': paths to any directories of patients to save. for example- glob("Training/HGG/**")
            (2) string 'type': options = reg (non-normalized), norm (normalized, but no bias correction), n4 (bias corrected and normalized)
    saves strips of patient slices to approriate directory (Training_PNG/, Norm_PNG/ or n4_PNG/) as patient-num_slice-num
    '''
    for patient_num, pat_id in enumerate(patients):
        print("***working on %d / %d patients***: %s" %
              (patient_num+1, len(patients), pat_id))
        #gt = glob(path + '/*more*/*.mha')
        # if glob(path + '/*GlistrBoost_ManuallyCorrected.nii.gz'):
        path = os.path.join(imgDir, pat_id)
        n4itk = True
        n4itk_apply = False
        zscore = True
        a = BrainPipeline(path, n4itk, n4itk_apply, bGT)
        if zscore == False:
            a.save_patient(type, patient_num, outDir)
        else:
            a.save_zscore(type, patient_num, outDir)


def createPatientList(path, nameList):

    patients = glob(path + '/**')
    if os.path.isfile(nameList):  # remove the existed file,
        os.remove(nameList)

    fObj = open(nameList, 'w+')  # create a new file for storing name list
    for pid in patients:
        if path in pid:
            id = pid.replace(path, '')
            fObj.write(id + '\n')
    fObj.close()
    return 0


if __name__ == '__main__':
    startTime = time.time()

    imgDir = '/home/linmin/Desktop/brats2019/all_training'
    outDir = '/home/linmin/Desktop/brats2019/zscore_training'
    # nameList = 'brats18_validation_list.txt'
    # bValue = createPatientList(imgDir, nameList)  # get patient IDs
    # patients = glob(imgDir + '/**')  # get full path of all patient
    patients = sorted(os.listdir(imgDir))
    bGT = 1  # with ground truth or not

    # patients = glob('/home/linmin/Desktop/temp/**')
    # h5f = h5py.File('patient_sequence.h5', 'w')
    # patients = glob('../BraTS_2018_Data_Validation/Data/**')
    # h5f = h5py.File('../BraTS_2018_Data_Validation/BRATS_18_Validation_patient_sequence.h5', 'w')
    # h5f.create_dataset('name_sequence', data=patients)
    # h5f.close()

    # save_labels(patients) # save labels of the patients
    # perform N4ITK bias correction and then apply normalization on the patients
    save_patient_slices(patients, 'n4', bGT, imgDir, outDir)
    endTime = time.time()
    print('******* It takes %d seconds to complete*********' %
          (endTime - startTime))
