from Dataloader import DataLoader
import torch
import sys
import torch.nn as nn
import time
import numpy as np
import cv2
import os
from SSIM import SSIM
from MMD_Loss import calculate_mmd
from model import Autoencodermodel
epochs = 150

ngpu = torch.cuda.device_count()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = Autoencodermodel()
model_name = 'AE-CFE-'

if ngpu > 1:
    model = nn.DataParallel(model)

model = model.to(device)



## Load the dataset
dataset = DataLoader()
traindataloader = torch.utils.data.DataLoader(dataset, batch_size=max(500 * ngpu,2), shuffle=True, num_workers=6)  ##32

optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()
criterion_1 = SSIM(window_size=10, size_average=True)




for epoch in range(epochs):
    loss = 0
    acc_imrec_loss = 0
    acc_featrec_loss = 0
    acc_mmdloss1 = 0
    acc_mmdloss2 = 0
    model.train()


    for feat, scimg,label,ds,_ in traindataloader:
        feat = feat.float()
        scimg = scimg.float()


        feat, scimg, ds = feat.to(device), scimg.to(device), ds.to(device)

        optimizer.zero_grad()

        z, output, im_out = model(feat)
        im_out = im_out.squeeze()

        feat_rec_loss = criterion(output, feat)
        imrec_loss = 1 - criterion_1(im_out,scimg)
        mmd_loss1, mmd_loss2 = calculate_mmd(z, ds)
        train_loss = imrec_loss + feat_rec_loss + 5*(mmd_loss1 + mmd_loss2)

        train_loss.backward()
        optimizer.step()

        loss += train_loss.data.cpu()
        acc_featrec_loss += feat_rec_loss.data.cpu()
        acc_imrec_loss += imrec_loss.data.cpu()
        acc_mmdloss1+=mmd_loss1.data.cpu()
        acc_mmdloss2+=mmd_loss2.data.cpu()

    loss = loss / len(traindataloader)
    acc_featrec_loss = acc_featrec_loss / len(traindataloader)
    acc_imrec_loss = acc_imrec_loss / len(traindataloader)
    acc_mmdloss1 = acc_mmdloss1/ len(traindataloader)
    acc_mmdloss2 = acc_mmdloss2/ len(traindataloader)
    # display the epoch training loss
    print("epoch : {}/{}, loss = {:.6f}, feat_loss = {:.6f},imrec_loss = {:.6f}, mmd_loss_mll = {:.6f},mmd_loss_pcb = {:.6f}".format
          (epoch + 1, epochs, loss, acc_featrec_loss, acc_imrec_loss, acc_mmdloss1, acc_mmdloss2))

    model.eval()
    for i in range(50):
        ft, img, lbl, _ ,_= dataset[i]
        ft = np.expand_dims(ft,axis=0)
        ft = torch.tensor(ft)
        ft = ft.to(device)

        _, _, im_out = model(ft)
        im_out = im_out.data.cpu().numpy() 
        im_out = np.squeeze(im_out)
        im_out = np.moveaxis(im_out,0,2)
        img = np.moveaxis(img,0,2)
        im = np.concatenate([img,im_out],axis=1)

        if epoch % 10 == 0:
            file_name = "out-images/"
            if os.path.exists(os.path.join(file_name)) is False:
                os.makedirs(os.path.join(file_name))
            cv2.imwrite(os.path.join(file_name, str(i) + "-" + str(epoch) + ".jpg"), im*255)

if os.path.exists(os.path.join('Model/')) is False:
        os.makedirs(os.path.join('Model/'))
torch.save(model, "Model/" + model_name+ time.strftime("%Y%m%d-%H%M%S") + ".mdl")

