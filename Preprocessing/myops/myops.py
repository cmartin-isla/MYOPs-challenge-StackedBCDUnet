# -*- coding: utf-8 -*-
# Authors:
# Víctor M. Campello (vicmancr@gmail.com)
# Carlos Martín-Isla (carlos.martin.isla.89@gmail.com) 

import os
import re
import glob
import pandas as pd

wd = 'D:/myops_unet/data/myops'
class Myops(object):
    '''
    Class for handling data files from UKBiobank dataset.
    '''
    def __init__(self, mode='training'):
        '''Constructor.'''
        self.base_path  = wd
        self.dataset    = 'myops'
        self.mode = mode


    def get_files(self):
        '''Obtain all file paths for each of the corresponding sets (training and testing).'''

        return self.get_files_in_folder(self.base_path)

    def get_files_in_folder(self, folder):
        '''
        Process all files inside given folder and extract available information.
        Returns:
            returns a list of files with keys info, image, mask, phase, patient_id
        '''
        file_list = []


        


        for f in sorted(os.listdir(self.base_path)):
            
            if 'DE' not in f:
                continue
            
            feid = int(f.split('_')[2])
            
            f_DE = os.path.join(self.base_path,f)

            mask = f_DE.replace('DE','gd')
            f_C0 = f_DE.replace('DE','C0')
            f_T2 = f_DE.replace('DE','T2')


            new_file = {'image_T2': f_T2,'image_C0': f_C0,'image_DE': f_DE, 'mask': mask, 'info': {'type': '2D'  }, 'patient_id': feid, 'phase': 'ED'}
            file_list.append(new_file)
            print(new_file)
        return file_list
