'''
Data augmentation requires numpy, random, rand scipy.ndimage library
Author: Linmin
Date: Apr_24, 2019
'''
import numpy as np
import random
from scipy.ndimage import rotate
from skimage.transform import rescale


class RandomRotation(object):
    """
    The data is zipped with tag: "image" and "labels"
    Random rotate the image patch and its label
    image:  nChannel*nRow*nCol*nSlice
    label:  nRow*nCol*nSlice
    output: rotated same size image and its label
    """

    def __init__(self, nMinimumDegree=0, nMaximumDegree=360):
        self.nMinimumDegree = nMinimumDegree
        self.nMaximumDegree = nMaximumDegree

    def random_rotation(self, data, degree):
        new_data = np.zeros(data.shape)
        cropx, cropy = data.shape[1], data.shape[2]
        startx, starty = 0, 0
        for idx in range(len(data)):
            temp = data[idx]
            temp_rot = rotate(temp, degree)
            y, x, z = temp_rot.shape
            startx = x // 2 - (cropx // 2)
            starty = y // 2 - (cropy // 2)
            new_temp = temp_rot[starty:starty+cropy, startx:startx+cropx, :]
            new_data[idx] = new_temp
        return new_data

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['labels']
        degree = random.randint(self.nMinimumDegree, self.nMaximumDegree)
        new_image = self.random_rotation(image, degree)
        new_landmarks = self.random_rotation(landmarks, degree)
        return {'image': new_image, 'labels': new_landmarks}


class scaling(object):
    """"
    scaling operation
    nMin and nMax are suggested to be around 1
    then resize the scaled image to original shape
    data.shape=nChannel, nRow, nColumn, nSlice
    label.shape=nChannel, nRow, nColumn, nSlice
    """

    def __init__(self, nMinValue=0.9, nMaxValue=1.1):
        self.nMinValue = nMinValue
        self.nMaxValue = nMaxValue

    # def center_crop(self, data, nRow, nCol, nSlice):
    #     nX, nY, nZ = data.shape
    #     if nX > nRow:
    #         start_x = (nX-nRow)//2
    #         start_y = (nY-nCol)//2
    #         start_z = (nSlice-nZ)//2
    #         new_data = data[start_x:start_x+nRow,
    #                         start_y: start_y+nCol, start_z:start_z+nSlice]
    #     else:
    #         start_x = (nRow-nX)//2
    #         start_y = (nCol-nY)//2
    #         start_z = (nSlice-nZ)//2
    #         new_data = np.zeros((nRow, nCol, nSlice))
    #         new_data[start_x:start_x+nX, start_y:start_y +
    #                  nY, start_z:start_z+nZ] = data

    #     return new_data

    def scaling_operation(self, data, nFactor=1):
        # 3D data only
        # nRow, nCol, nSlice = data.shape[-3], data.shape[-2], data.shape[-1]
        # nRow, nCol = data.shape[-2], data.shape[-1] #2D data only
        new_data = []
        for idx in range(len(data)):
            temp = data[idx]
            temp = rescale(temp, nFactor, mode='reflect',
                           multichannel=True, anti_aliasing=False)
            nX, nY, nZ = temp.shape[0], temp.shape[1], temp.shape[2]
            if len(new_data) == 0:
                new_data = np.zeros((len(data), nX, nY, nZ))
            # temp = self.center_crop(temp, nRow, nCol, nSlice)
            new_data[idx] = temp
        return new_data

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['labels']
        nFactor = random.uniform(self.nMinValue, self.nMaxValue)
        new_image = self.scaling_operation(image, nFactor)
        new_landmarks = self.scaling_operation(landmarks, nFactor)
        return {'image': new_image, 'labels': new_landmarks}


class centerCrop(object):
    """"
    scaling operation
    nMin and nMax are suggested to be around 1
    then resize the scaled image to original shape
    data.shape=nChannel, nRow, nColumn, nSlice
    label.shape=nChannel, nRow, nColumn, nSlice
    """

    def __init__(self, nRow, nCol, nSlice):
        self.nRow = nRow
        self.nCol = nCol
        self.nSlice = nSlice

    def center_crop(self, data):
        nC, nX, nY, nZ = data.shape[0], data.shape[1], data.shape[2], data.shape[3]
        nRow, nCol, nSlice = self.nRow, self.nCol, self.nSlice

        if nX > nRow:
            start_x = (nX-nRow)//2
            start_y = (nY-nCol)//2
            start_z = (nSlice-nZ)//2
            new_data = data[:, start_x:start_x+nRow,
                            start_y: start_y+nCol, start_z:start_z+nSlice]
        else:
            start_x = (nRow-nX)//2
            start_y = (nCol-nY)//2
            start_z = (nSlice-nZ)//2
            new_data = np.zeros((nC, nRow, nCol, nSlice))
            new_data[:, start_x:start_x+nX, start_y:start_y +
                     nY, start_z:start_z+nZ] = data

        return new_data

    def __call__(self, sample):
        image, landmarks = sample['image'], sample['labels']
        new_image = self.center_crop(image)
        new_landmarks = self.center_crop(landmarks)
        return {'image': new_image, 'labels': new_landmarks}
