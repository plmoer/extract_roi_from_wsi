'''
For BraTS 2018 training and validation.
after normalization image within 0 and 1, the mean of flair, t1, t1ce, t2 is [0.4484 0.4515 0.4502 0.4337]
the std of flair, t1, t1ce, t2 is [0.1923 0.1633 0.1675 0.2183]

'''
import torch
from torch.utils.data.dataset import Dataset
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
import os
from torchvision import transforms, utils
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import time
from utils.jpu_net import JPUNet
import pandas as pd
# from utils.unet_vae import UNet
from scipy.ndimage import rotate
import nibabel as nib
import random
from skimage.transform import rescale, resize
import matplotlib.pyplot as plt
import csv
from scipy import ndimage


startTime = time.time()

rootDir = os.getcwd()
train_dataFolderDir = os.path.join(
    rootDir, 'last_vae_309_uncertainty_orignal')  # path to all patches
valid_dataFolderDir = os.path.join(
    rootDir, 'last_vae_309_uncertainty_orignal')  # path to all patches

train_info = os.path.join(
    rootDir, 'training_data_classification_labels_3risk.csv')
# valid_info = os.path.join(rootDir, 'survival_evaluation.csv')

model_path = os.path.join(rootDir, 'models_mri_all/')
if os.path.isdir(model_path) == False:
    os.mkdir(model_path)

BATCH_SIZE = 1
nTotalEpoch = 400
nStep = 400  # save model at every 3 iteration
in_channel = 3

nRow, nCol, nSlice = 160, 192, 128
ndf = 4
LR = 0.001
nClass = 3

outTextDir = os.path.join(rootDir, 'accu.txt')
if os.path.exists(outTextDir):
    dFile = open(outTextDir, 'a')  # apend the image into existing txt file
else:
    dFile = open(outTextDir, 'w')  # save the image into txt file
dFile.write("--------------*************----------------\n")
dFile.write(""+"Epoch" + "\t" + "T_Loss_old" + "\t" + "T_Loss_new" + "\n")


class DriveData(Dataset):
    __xs = []  # image
    __ys = []  # image

    def __init__(self, dataFolderPath, list_info, pat_info,  transform=None):
        self.dataFolderPath = dataFolderPath
        self.list_info = list_info
        self.pat_info = pat_info
        self.nClass = nClass
        self.myTransform = transform
        self.information = self.getPatient(pat_info)
        # print(self.information)
        with open(self.list_info) as f:
            for idx, line in enumerate(f):
                # get fused modality image name
                pid = line.split()[0]
                risk = self.information[pid]

                self.__xs.append(line.split()[0])
                self.__ys.append(risk)

    def getPatient(self, pat_info):
        dCSV = pd.read_csv(pat_info)
        tempPatientList = dCSV.CPM_RadPath_2019_ID  # only take training patient list
        # tempAgeList = dCSV.age_in_days
        all_patientList = tempPatientList.dropna()  # remove NaN from the column
        # ageList = tempAgeList.dropna()  # remove NaN from the column
        tempRiskList = dCSV.risk
        riskList = tempRiskList.dropna()  # remove NaN from the column
        info_list = {}
        for idx in range(0, len(all_patientList)):
            pat_id = all_patientList[idx]
            # age_id = ageList[idx]
            if len(riskList) == 0:
                nRisk = 0
            else:
                nRisk = riskList[idx]
            temp = {pat_id: nRisk}
            if len(info_list) == 0:
                info_list = temp
            else:
                info_list.update(temp)
        return info_list

    def __getitem__(self, index):
        # print('------------: ', index)
        data_path = os.path.join(
            self.dataFolderPath, self.__xs[index]+'_all_uncertainty.nii.gz')
        features = nib.load(data_path).get_data()
        nRisk = torch.from_numpy(np.array(self.__ys[index]).reshape([1]))
        if self.myTransform is not None:
            img = self.myTransform(img)
        return features, nRisk

    # Override to give PyTorch size of dataset
    def __len__(self):
        return len(self.__xs)  # -*- coding: utf-8 -*-


transformations_train = None

trainPatchInfoPath = os.path.join(rootDir, 'trainList_all.txt')
train_dataset = DriveData(
    train_dataFolderDir, trainPatchInfoPath,  train_info,  transformations_train)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE, shuffle=True, num_workers=32)


# transformations_validation = None
# validationPatchInfoPath = os.path.join(rootDir, 'validList.txt')
# validation_dataset = DriveData(
#     valid_dataFolderDir, validationPatchInfoPath,  valid_info,  transformations_validation)
# validation_loader = DataLoader(
#     dataset=validation_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=32)

model = JPUNet(in_channel, nClass, ndf)
model = model.cuda()
optimizer = optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()


def train():
    model.train()
    train_loss = 0
    train_dice = 0
    nCorrectCount = 0
    nTotalCount = 0
    factor = 0.001
    for batch_idx, (data, target) in enumerate(train_loader):
        # index = (data == 0)  # find index for background
        data = data.type(torch.FloatTensor)
        data = data/100
        # data = data.view(data.size(0), in_channel, nX, nY, nZ)
        # data = data + 0.01*torch.randn_like(data)  # add noise
        # data[index] = 0
        target = target.type(torch.LongTensor)
        target = target.view(-1)

        data, target = Variable(data).cuda(), Variable(
            target).cuda()  # gpu version
        optimizer.zero_grad()

        # showImage(data, target)  # show image
        output, seLoss = model(data)
        loss = criterion(output, target)
        reg_loss = None
        for param in model.parameters():
            if reg_loss is None:
                reg_loss = param.norm(2)
            else:
                reg_loss = reg_loss + param.norm(2)

        _, predicted = torch.max(output.data, 1)

        if predicted == target:
            nCorrectCount = nCorrectCount + 1
        nTotalCount = nTotalCount + 1

        loss = loss + torch.sum(torch.abs(seLoss)) + factor*reg_loss
        train_loss = train_loss+loss.data.cpu().numpy()

        loss.backward()
        optimizer.step()
    train_dice = nCorrectCount/nTotalCount
    print('.....Average training loss: %.4f, dice: %.4f' %
          (train_loss/batch_idx, train_dice))
    return train_loss/(batch_idx + 1), train_dice


def adjust_learning_rate(optimizer, LR, epoch):
    for param_group in optimizer.param_groups:
        lr = param_group['lr']
        lr = LR * ((1-epoch/nTotalEpoch).__pow__(0.9))
        param_group['lr'] = lr
        print('......LR is %f at epoch %d.' % (param_group['lr'], epoch+1))


def computeRisk(target):
    aMin = 31*10
    aMax = 31*15
    if target > aMax:
        risk = 3
    elif target < aMin:
        risk = 1
    else:
        risk = 2
    return risk


def test():
    model.eval()
    nCorrectCount = 0
    nTotalCount = 0
    valid_loss = 0
    valid_dice = 0
    for batch_idx,  (data, age, target) in enumerate(validation_loader):
        #data = image['image']
        #target = image['labels']
        data = data.type(torch.FloatTensor)
        age = age.type(torch.FloatTensor)
        target = target.type(torch.FloatTensor)
        # data = data / 255.0
        # data = (data-0.5)/0.5
        # data[0:BATCH_SIZE, 0] = (data[0:BATCH_SIZE, 0] - 0.4484) / 0.1923
        # data[0:BATCH_SIZE, 1] = (data[0:BATCH_SIZE, 1] - 0.4515) / 0.1633
        # data[0:BATCH_SIZE, 2] = (data[0:BATCH_SIZE, 2] - 0.4502) / 0.1675
        # data[0:BATCH_SIZE, 3] = (data[0:BATCH_SIZE, 3] - 0.4337) / 0.2183
        data, age, target = Variable(data).cuda(), Variable(age).cuda(), Variable(
            target).cuda()  # gpu version
        output = model(data, age)
        tar_value = target.data.cpu().numpy()
        out_value = output.data.cpu().numpy().astype(int)
        tar_risk = computeRisk(tar_value)
        out_risk = computeRisk(out_value)
        if tar_risk == out_risk:
            nCorrectCount = nCorrectCount + 1
        nTotalCount = nTotalCount + 1
        valid_loss = valid_loss + np.absolute(tar_value - out_value)
        print('..........: target %d prediction: %d' % (tar_value, out_value))
    valid_ave_loss = valid_loss/(batch_idx + 1)
    valid_dice = nCorrectCount/nTotalCount
    print('... Average validation acc: %.4f' % (valid_dice))
    return valid_dice


nLoss = 1e15
nDice = 0
modelPath_stat = model_path+'model_checkpoint_'
for epoch in range(0, nTotalEpoch):
    print('\n......Working on the %d epoch out of %d.' %
          (epoch+1, nTotalEpoch))
    # scheduler.step()
    adjust_learning_rate(optimizer, LR, epoch)  # adjust lr
    train_ave_loss, train_ave_dice = train()
    # valid_ave_dice = test()
    #   valid_dice_result, ' and old accuracy: ', nDice)
    # dFile.write(str(epoch) + "\t" + str(np.round(nDice, 4))+"\t\t" +
    #             str(np.round(valid_ave_dice, 4)) + "\t" + "\n")

    print('=====>The old train loss: %.4f new loss: %.4f, ....., old train dice: %.4f and new train dice: %.4f' %
          (nLoss, train_ave_loss, nDice, train_ave_dice))
    if train_ave_loss < nLoss:
        nLoss = train_ave_loss  # update the validation accuracy
        print('*********The updated average train loss: %.4f, dice: %.4f' %
              (train_ave_loss, train_ave_dice))
        torch.save(model, os.path.join(
            model_path, 'bestModel_' + str(epoch)))
    # elif np.mod(epoch, nStep) == 0:
    #     torch.save({
    #         'epoch': epoch,
    #         'model_state_dict': model.state_dict(),
    #         'optimizer_state_dict': optimizer.state_dict()
    #     }, os.path.join(model_path, 'model_checkpoint_' + tumor_type+'_'+view_type+'_'+str(bVAE)+'_'+str(epoch)))

endTime = time.time()
print('******* It takes %d seconds to complete*********' % (endTime-startTime))
