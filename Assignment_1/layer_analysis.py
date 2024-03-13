import cv2 as cv
import numpy as np
import os
import cca
import pandas as pd
import calc_dice

KER, EPI, DRM, DEJ = [224, 224, 224], [160, 48, 112], [0, 255, 190], [255, 172, 255] #in BGR format\

# image = cv.imread('Dataset/Train/Tissue/RA23-01882-A1-1-PAS.[5120x1024].jpg', 0)
for i in range(1, 26): 
    filenum = i
    image = cv.imread(f'Dataset/Train/Tissue/{filenum}.jpg', 0)
    # image = cv.imread('Dataset/Train/Tissue/RA23-01882-A1-1-PAS.[5120x512].jpg', 0)
    image = cca.extractBackground(image)
    # cv.imshow('og', image)
    # cv.waitKey()

    #########################################################

    df = pd.read_csv('Mode/mode_table.csv',sep=',')
    df2 = pd.read_csv('Mode/mode_table2.csv',sep=',')
    df3 = pd.read_csv('Mode/mode_table3.csv',sep=',')
    df4 = pd.read_csv('Mode/mode_table4.csv',sep=',')

    #########################################################

    kerrange = df.loc[:, 'KER'].tolist()
    kerrange.extend(df2.loc[:, 'KER'].tolist())
    kerrange.extend(df3.loc[:, 'KER'].tolist())
    kerrange.extend(df4.loc[:, 'KER'].tolist())
    kerrange = list(set(kerrange))
    # kerrange.extend(range(40,90,1))
    # kerrange.extend(range(250,255,1))
    imageKER = cca.CCAwithMode(image, kerrange, KER)
    # cv.imshow('KER', imageKER)
    # cv.waitKey()

    #########################################################

    epirange = df.loc[:, 'EPI'].tolist()
    epirange.extend(df2.loc[:, 'EPI'].tolist())
    epirange.extend(df3.loc[:, 'EPI'].tolist())
    epirange.extend(df4.loc[:, 'EPI'].tolist())
    epirange = list(set(epirange))
    # epirange.extend(range(160,180,1))
    imageEPI = cca.CCAwithMode(image, epirange,EPI)
    # cv.imshow('EPI', imageEPI)
    # cv.waitKey()
    height, width, channels = imageKER.shape
    for x in range(0,height):
            for y in range(0,width):
                if np.array_equal(imageEPI[x][y], [0,0,0]):
                    continue
                else:
                    imageKER[x][y] = imageEPI[x][y]

    # cv.imshow('EPI', imageKER)
    # cv.waitKey()

    #########################################################

    drmrange = df.loc[:, 'DRM'].tolist()
    drmrange.extend(df2.loc[:, 'DRM'].tolist())
    drmrange.extend(df3.loc[:, 'DRM'].tolist())
    drmrange.extend(df4.loc[:, 'DRM'].tolist())
    drmrange = list(set(drmrange))
    # drmrange.extend(range(240,250,1))
    imageDRM = cca.CCAwithMode(image, drmrange,DRM)
    # cv.imshow('DRM', imageDRM)
    # cv.waitKey()
    height, width, channels = imageKER.shape
    for x in range(0,height):
            for y in range(0,width):
                if np.array_equal(imageDRM[x][y], [0,0,0]):
                    continue
                else:
                    imageKER[x][y] = imageDRM[x][y]

    # cv.imshow('EPI', imageKER)
    # cv.waitKey()

    # dejrange = []

    # dejrange.extend(range(200,210,1))
    dejrange = df3.loc[:, 'DEJ'].tolist()
    # dejrange.extend(df2.loc[:, 'DEJ'].tolist())
    # dejrange.extend(df.loc[:, 'DEJ'].tolist())
    dejrange.extend(df4.loc[:, 'DEJ'].tolist())
    dejrange = list(set(dejrange))
    imageDEJ = cca.CCAwithMode(image, dejrange,DEJ)
    # cv.imshow('DEJ', imageDEJ)
    # cv.waitKey()
    height, width, channels = imageKER.shape
    for x in range(0,height):
            for y in range(0,width):
                if np.array_equal(imageDEJ[x][y], [0,0,0]):
                    continue
                else:
                    imageKER[x][y] = imageDEJ[x][y]

    # cv.imshow('EPI', imageKER)
    # cv.waitKey()

    cv.imwrite(f'Dataset/Train/Output/{filenum}.png',imageKER)

    print(f'Dice (file {filenum}) : {calc_dice.dice(filenum)}')