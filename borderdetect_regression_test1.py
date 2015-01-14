# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 15:28:53 2015

@author: mzhang

detect borders of LP by regression Neuro Network
"""


import numpy as np
import lpfunctions as lpfuncs
import functions as funcs
import cv2 as cv
import cPickle
import time
from scipy.cluster.vq import kmeans
import regNN

maxiter = 10
neednum = 8000
hmmgmmfile = 'borderdetect(2015.01.09.01)_' + str(neednum) + '_.bin'
folderpath = '/Users/mzhang/work/LPR_TrainData/new/'
neednum2 = 0
folderpath2 = '/Users/mzhang/work/LPR_TrainData/old/'
stdshape1 = (28, 14) #(28, 32) #(28, 14)
#neednum = 100
trainnum = neednum + neednum2 # - 4000
teststart = 0 #neednum - 4000

stdheight = 64
gwndlen = 24
bTrain = False


def getNormalizedHist(colpart):
    sbxhist = np.sum(colpart, axis=1)
    minv = np.min(sbxhist)
    maxv = np.max(sbxhist)
    nhist = (sbxhist - minv) * 1.0 / (maxv - minv + 1)
    
    return nhist


def histAnalyze(ys):
    ynum, ydim = ys.shape
    y1 = ys[:, 0]
    y2 = ys[:, 1]
    hist1 = np.zeros(stdheight, dtype=np.int32)
    hist2 = np.zeros(stdheight, dtype=np.int32)
    for yi in xrange(ynum):
        if y1[yi] > 0 and y1[yi] < stdheight:
            hist1[y1[yi]] += 1
        if y2[yi] > 0 and y2[yi] < stdheight:
            hist2[y2[yi]] += 1
    
#    print hist1
#    print hist2
    mainy1 = np.argmax(hist1)
    mainy2 = np.argmax(hist2)
    
    return mainy1, mainy2
    

def getTrainingDatas(lpinfo_list):
    print 'get training data....'
    
    samplexlist = []
    sampleylist = []
    wndlen = gwndlen
    for lpi in lpinfo_list:
        gimg = cv.imread(lpi.img_fn, cv.CV_LOAD_IMAGE_GRAYSCALE)
#        chbbs = lpi.char_bndbox
        bbs = lpi.lp_bndbox
        margin1 = np.random.randint(0, (bbs[3]-bbs[1]))
        margin2 = np.random.randint(0, (bbs[3]-bbs[1]))
        marginw = (bbs[2]-bbs[0]) / 2
        bbs2 = [np.max([0, bbs[0]-marginw]), np.max([0, bbs[1]-margin1]), 
                np.min([gimg.shape[1]-1, bbs[2]+marginw]), np.min([gimg.shape[0]-1, bbs[3]+margin2])]
        lpimg = np.copy(gimg[bbs2[1]:bbs2[3]+1, bbs2[0]:bbs2[2]+1])
        newbbs = [bbs[0]-bbs2[0], bbs[1]-bbs2[1], bbs[2]-bbs2[0], bbs[3]-bbs2[1]]
        
        
        rszheight = stdheight
        rszrate = rszheight * 1. / lpimg.shape[0]
        rszwidth = int(lpimg.shape[1] * rszrate)
        lpimg = cv.resize(lpimg, (rszwidth, rszheight))        
        sblx = funcs.siSobelX_U8(lpimg)
        
        for ni in xrange(4):
            newbbs[ni] = int(newbbs[ni] * rszrate)
        
        y0 = newbbs[1] * 1. / stdheight
        y1 = newbbs[3] * 1. / stdheight
        for ci in xrange(newbbs[0], newbbs[2]-wndlen, 4):
            colpart = sblx[:, ci:ci+wndlen]
            nhist = getNormalizedHist(colpart)
            samplexlist.append(nhist)
            sampleylist.append([y0, y1])
        
#        print samplexlist
#        print sampleylist
        if 0:
            tmpgimg = sblx
            clpimg = np.zeros((lpimg.shape[0], lpimg.shape[1], 3), dtype=np.uint8)
            clpimg[:, :, 0] = tmpgimg
            clpimg[:, :, 1] = tmpgimg
            clpimg[:, :, 2] = tmpgimg
            cv.rectangle(clpimg, (newbbs[0], newbbs[1]), (newbbs[2], newbbs[3]), [0, 255, 0])
            cv.imshow('lpimg', clpimg)
            cv.waitKey(10)
            
    x = np.asarray(samplexlist, dtype=np.float32)
    y = np.asarray(sampleylist, dtype=np.float32)
    
    return x, y


def train(lpinfo_list):
    maxsampling = 100
    for li in xrange(maxsampling):
        print '----------[%d/%d]----------'%(li+1, maxsampling)
        x, y = getTrainingDatas(lpinfo_list)
        print 'sample size:', x.shape, y.shape
        regNN.train(x, y, 100)


def test(lpinfo_list):
    wndlen = gwndlen
    for lpi in lpinfo_list:
        gimg = cv.imread(lpi.img_fn, cv.CV_LOAD_IMAGE_GRAYSCALE)
#        chbbs = lpi.char_bndbox
        bbs = lpi.lp_bndbox
        margin1 = np.random.randint(0, (bbs[3]-bbs[1]))
        margin2 = np.random.randint(0, (bbs[3]-bbs[1]))
        marginw = (bbs[2]-bbs[0]) / 2
        bbs2 = [np.max([0, bbs[0]-marginw]), np.max([0, bbs[1]-margin1]), 
                np.min([gimg.shape[1]-1, bbs[2]+marginw]), np.min([gimg.shape[0]-1, bbs[3]+margin2])]
        lpimg = np.copy(gimg[bbs2[1]:bbs2[3]+1, bbs2[0]:bbs2[2]+1])
        newbbs = [bbs[0]-bbs2[0], bbs[1]-bbs2[1], bbs[2]-bbs2[0], bbs[3]-bbs2[1]]
        
        
        rszheight = stdheight
        rszrate = rszheight * 1. / lpimg.shape[0]
        rszwidth = int(lpimg.shape[1] * rszrate)
        lpimg = cv.resize(lpimg, (rszwidth, rszheight))        
        sblx = funcs.siSobelX_U8(lpimg)
        
        for ni in xrange(4):
            newbbs[ni] = int(newbbs[ni] * rszrate)
        
        samplexlist = []
        for ci in xrange(0, lpimg.shape[1]-wndlen, 1):
            colpart = sblx[:, ci:ci+wndlen]
            nhist = getNormalizedHist(colpart)
            samplexlist.append(nhist)
        
        x = np.asarray(samplexlist, dtype=np.float32)
        y = regNN.test(x)
        realy = y * stdheight + 0.5
        realy = realy.astype(np.int32)
        meany = np.mean(realy, axis=0) + 0.5
        meany = meany.astype(np.int32)
        print meany
        
        meany = histAnalyze(realy)
        
        if 1:
            tmpgimg = lpimg
            clpimg = np.zeros((lpimg.shape[0], lpimg.shape[1], 3), dtype=np.uint8)
            clpimg[:, :, 0] = tmpgimg
            clpimg[:, :, 1] = tmpgimg
            clpimg[:, :, 2] = tmpgimg
#            cv.rectangle(clpimg, (newbbs[0], newbbs[1]), (newbbs[2], newbbs[3]), [0, 255, 0])
            cv.line(clpimg, (0, meany[0]), (lpimg.shape[1]-1, meany[0]), [0, 0, 255])
            cv.line(clpimg, (0, meany[1]), (lpimg.shape[1]-1, meany[1]), [0, 0, 255])
            
            cv.imshow('lpimg', clpimg)
            cv.waitKey(0)

def test2():
    fpath = '/Users/mzhang/Downloads/'
    fnlist = ['1 (14).jpg', '1 (16).jpg', '1 (20).jpg', '1.jpg', '2.jpg', '3.jpg', '4.jpg']
    wndlen = gwndlen
    for fni in fnlist:
        imgfn = fpath + fni
        lpimg = cv.imread(imgfn, cv.CV_LOAD_IMAGE_GRAYSCALE)

        rszheight = stdheight
        rszrate = rszheight * 1. / lpimg.shape[0]
        rszwidth = int(lpimg.shape[1] * rszrate)
        lpimg = cv.resize(lpimg, (rszwidth, rszheight))
        sblx = funcs.siSobelX_U8(lpimg)

        samplexlist = []
        for ci in xrange(0, lpimg.shape[1], 1):
            colpart = sblx[:, ci:ci+wndlen]
            nhist = getNormalizedHist(colpart)
            samplexlist.append(nhist)
        
        x = np.asarray(samplexlist, dtype=np.float32)
        y = regNN.test(x)
        realy = y * stdheight + 0.5
        realy = realy.astype(np.int32)
        meany = np.mean(realy, axis=0) + 0.5
        meany = meany.astype(np.int32)
#        print realy
        meany = histAnalyze(realy)
        
        if 1:
            tmpgimg = lpimg
            clpimg = np.zeros((lpimg.shape[0], lpimg.shape[1], 3), dtype=np.uint8)
            clpimg[:, :, 0] = tmpgimg
            clpimg[:, :, 1] = tmpgimg
            clpimg[:, :, 2] = tmpgimg
#            cv.rectangle(clpimg, (newbbs[0], newbbs[1]), (newbbs[2], newbbs[3]), [0, 255, 0])
            cv.line(clpimg, (0, meany[0]), (lpimg.shape[1]-1, meany[0]), [0, 0, 255])
            cv.line(clpimg, (0, meany[1]), (lpimg.shape[1]-1, meany[1]), [0, 0, 255])
            
            cv.imshow('lpimg', clpimg)
            cv.waitKey(0)

if bTrain:
    cost_times = {}
    t1 = time.time()
    lpinfo_list2 = lpfuncs.getall_lps3(folderpath2, neednum2)
    lpinfo_list = lpfuncs.getall_lps3(folderpath, neednum)
    lpinfo_list = lpinfo_list + lpinfo_list2
    t2 = time.time()
    cost_times['getall_lps2'] = t2-t1
    print 'total sample number:', len(lpinfo_list)
    
    train(lpinfo_list)

elif 0:
    cost_times = {}
    t1 = time.time()
    neednum = 1000
    lpinfo_list = lpfuncs.getall_lps3(folderpath, neednum)
    lpinfo_list = lpinfo_list
    t2 = time.time()
    cost_times['getall_lps2'] = t2-t1
    print 'total sample number:', len(lpinfo_list)
    
    test(lpinfo_list)

else:
    test2()

