import numpy as np
import cv2 as cv
import cca
import pandas as pd
import os
import time


start_time = time.time()
KER, EPI, DRM, DEJ = [224, 224, 224], [160, 48, 112], [0, 255, 190], [255, 172, 255] #in BGR format\
KER_l, EPI_l, DRM_l, DEJ_l = [],[],[],[]
pathtissue = 'Dataset/Train/Tissue'
pathmask = 'Dataset/Train/masks'
# has list of all tissue images
tissue_list = os.listdir(pathtissue)
tissue_list.sort()
mask_list = os.listdir(pathmask) 
mask_list.sort()

# print(type(tissue_list)) 

if len(mask_list) == len(tissue_list):
    for i in range(0,len(mask_list)):
    # for i in range(0,1):
        im_bw = cv.imread((pathtissue+'/'+tissue_list[i]), 0)
        height, width = im_bw.shape
        im_mask = cv.imread(pathmask+'/'+mask_list[i])
        # cv.imshow('Input',im_bw)
        # cv.imshow('Output',im_res)
        # compare both pictures to find out the data
        KER_l, EPI_l, DRM_l, DEJ_l = [],[],[],[]
        for x in range(0,height):
            for y in range(0,width):
                if np.array_equal(im_mask[x][y], EPI):
                    EPI_l.append(im_bw[x][y])
                elif np.array_equal(im_mask[x][y], KER):
                    KER_l.append(im_bw[x][y])
                elif np.array_equal(im_mask[x][y], DRM):
                    DRM_l.append(im_bw[x][y])
                elif np.array_equal(im_mask[x][y], DEJ):
                    DEJ_l.append(im_bw[x][y])
        
        max_length = max(len(KER_l), len(EPI_l), len(DRM_l), len(DEJ_l))

        # Pad the shorter lists with NaN values to make them the same length
        KER_l += [np.nan] * (max_length - len(KER_l))
        EPI_l += [np.nan] * (max_length - len(EPI_l))
        DRM_l += [np.nan] * (max_length - len(DRM_l))
        DEJ_l += [np.nan] * (max_length - len(DEJ_l))

        # Create a DataFrame
        df = pd.DataFrame({
            'KER': KER_l,
            'EPI': EPI_l,
            'DRM': DRM_l,
            'DEJ': DEJ_l
        })

        # Save DataFrame to Excel file
        df.to_csv('Excel_Data/output' + str(i) + '.csv', index=False)
        print('.')

# Get the maximum length of the lists
# max_length = max(len(KER_l), len(EPI_l), len(DRM_l), len(DEJ_l))

# # Pad the shorter lists with NaN values to make them the same length
# KER_l += [np.nan] * (max_length - len(KER_l))
# EPI_l += [np.nan] * (max_length - len(EPI_l))
# DRM_l += [np.nan] * (max_length - len(DRM_l))
# DEJ_l += [np.nan] * (max_length - len(DEJ_l))

# # Create a DataFrame
# df = pd.DataFrame({
#     'KER': KER_l,
#     'EPI': EPI_l,
#     'DRM': DRM_l,
#     'DEJ': DEJ_l
# })

# # Save DataFrame to Excel file
# df.to_csv('output.csv', index=False)


print("--- %s seconds ---" % (time.time() - start_time))

print(f'Min: {min(KER_l)}, {min(EPI_l)}, {min(DRM_l)}, {min(DEJ_l)}')
print(f'Max: {max(KER_l)}, {max(EPI_l)}, {max(DRM_l)}, {max(DEJ_l)}')
print(f'Mean: {np.nanmean(KER_l)}, {np.nanmean(EPI_l)}, {np.nanmean(DRM_l)}, {np.nanmean(DEJ_l)}')
            

