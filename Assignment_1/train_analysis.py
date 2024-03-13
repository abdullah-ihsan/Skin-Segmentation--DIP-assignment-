import cv2 as cv
import numpy as np
import os
import cca

path = 'Dataset/Train/Tissue'
# has list of all tissue images
dir_list = os.listdir(path) 

for sample in dir_list:
    im_bw = cv.imread(path+'/'+sample, 0)
    cv.imshow('Input',im_bw)
    cca.extractBackground(im_bw)

