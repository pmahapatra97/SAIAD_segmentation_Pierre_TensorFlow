# -*- coding: utf-8 -*-

from PIL import Image
# import data as dt
import constant
import matplotlib.pyplot as plt
import numpy as np
import os


# d = dt.SaiadData("tumeur","SAIAD_01")
# Isum_list=[]
# for patient in constant.BASE_CASE_PATIENT_NAMES:
#     Is, Ls = d.get_patient_slices([patient])
#     Isum = Is.squeeze().mean(axis=0)
#     Isum_list.append(Isum)

a = -0.001
ya = [0,a,a,a,a,a,a,0,a,0.5*a,a,a,a,a]
yc = [520,450,420,470,450,400,430,475,440,395,450,450,430,420]
xa = 35
xy = [-xa,xa,xa,-xa,-xa,xa,xa,xa,xa,xa,xa,int(xa*0.75),-xa,-xa]




xx = np.arange(0,512)
fig = plt.figure()
ax = fig.gca()
fig.show()

for i,patient in enumerate(constant.BASE_CASE_PATIENT_NAMES):
    listFile = [s for s in os.listdir(constant.SCANNER_DIR.format(patient)) if ".jpg" in s]
    path = constant.SCANNER_PREPROC_DIR.format(patient)
    if not os.path.exists(path):
        os.mkdir(path)
    listIdx = list(map(lambda file:int(os.path.splitext(file)[0]),listFile))
    listIdx.sort()
    yy = ya[i]*(xx-256)**2 + yc[i]
    xxyy = xy[i]
    for idx in listIdx:
        im = Image.open(constant.SCANNER_DIR.format(patient)+'{}.jpg'.format(idx)).convert('L')
        im_a = np.array(im)
        x, y = (im_a>=0).nonzero()
        x = x.reshape(512,512,order="F")
        y = y.reshape(512,512,order="F")
        yy = ya[i]*(xx-256)**2 + yc[i]
        xxyy = xy[i]
        aa,bb = np.logical_or(x<=xxyy,y>=yy).nonzero()
        im_bis = im_a.copy()
        im_bis[aa,bb] = 0
        im_b = Image.fromarray(im_bis)
        im_b.save(constant.SCANNER_PREPROC_DIR.format(patient)+'{}.jpg'.format(idx))

# Isum_arr = np.array(Isum_list).reshape(2,7,512,512)
# fig, axs = plt.subplots(*Isum_arr.shape[:2], figsize = (40,12))
# xx = np.arange(0,512)
# for i, l in enumerate(Isum_arr):
#     for j, Isum in enumerate(l):
#         pos = axs[i,j].imshow(np.array(Isum), cmap=plt.get_cmap("binary_r"),vmin=0,vmax=1)
#         axs[i,j].set_title(constant.BASE_CASE_PATIENT_NAMES[7*i+j],fontsize=30)
#         yy = 450 + np.zeros(xx.shape)
#         yy = ya[7*i+j]*(xx-256)**2 + yc[7*i+j]
#         #axs[i,j].plot(xx,yy, color="b")
#         xxyy = xy[7*i+j]
#         #axs[i,j].plot([xxyy,xxyy],[0,511], color ="r")
#         #fig.colorbar(pos, ax=axs[i,j])