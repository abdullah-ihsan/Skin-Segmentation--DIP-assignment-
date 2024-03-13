import numpy as np
import cv2 as cv
import cca
import pandas as pd
import os
import time


path = 'Excel_Data'
# has list of all tissue images
dir_list = os.listdir(path) 
KERmode, EPImode, DRMmode, DEJmode = [], [], [], []

for i in range(0,len(dir_list)-1):
# sample = 'output25.csv'
    df = pd.read_csv(path+'/output'+str(i)+'.csv', sep=',')
    # print(df)
    mode = df.loc[:, 'KER'].mode()
    freq = df.loc[:, 'KER'].value_counts()
    if len(freq) > 0: 
        s = freq.index[4]
        print(float(s), end=', ')
        KERmode.append(s)
    else: print('xxxxx', end=', ')
    mode = df.loc[:, 'EPI'].mode()
    freq = df.loc[:, 'EPI'].value_counts()
    if len(freq) > 0: 
        s = freq.index[4]
        print(float(s), end=', ')
        EPImode.append(s)
    else: print('xxxxx', end=', ')
    mode = df.loc[:, 'DRM'].mode()
    freq = df.loc[:, 'DRM'].value_counts()
    if len(freq) > 0: 
        s = freq.index[4]
        print(float(s), end=', ')
        DRMmode.append(s)
    else: print('xxxxx', end=', ')
    mode = df.loc[:, 'DEJ'].mode()
    freq = df.loc[:, 'DEJ'].value_counts()
    if len(freq) > 0: 
        s = freq.index[4]
        print(float(s), end=', ')
        DEJmode.append(s)
    else: print('xxxxx')

max_length = max(len(KERmode), len(EPImode), len(DRMmode), len(DEJmode))
KERmode += [np.nan] * (max_length - len(KERmode))
EPImode += [np.nan] * (max_length - len(EPImode))
DRMmode += [np.nan] * (max_length - len(DRMmode))
DEJmode += [np.nan] * (max_length - len(DEJmode))

df = pd.DataFrame({
    'KER': KERmode,
    'EPI': EPImode,
    'DRM': DRMmode,
    'DEJ': DEJmode,
})        

df.to_csv('mode_table4.csv', index=False)