# -*- coding: utf-8 -*-
"""
Created on Tue Dec  9 11:12:18 2014

@author: mzhang
"""

import lpfunctions as lpfuncs
import cv2 as cv
import simpleCNN as scnn


folderpath = '/Volumes/ZMData1/LPR_TrainData/new/'
neednum = 1
stdshape = (32, 14)

lpinfo_list = lpfuncs.getall_lps2(folderpath, neednum, ifstrech=False)

for pi, lp_one in enumerate(lpinfo_list):
#    gimg = lp_one.charobj.grayimg
#    fn = 'image/' + str(pi) + '.bmp'
#    cv.imwrite(fn, gimg)
    scnn.tmptest2(lp_one, stdsize=stdshape)
    break

