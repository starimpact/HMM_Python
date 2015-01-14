# -*- coding: utf-8 -*-
"""
Created on Sun Aug 17 18:10:30 2014

@author: Administrator
"""

import cv2
import os
import numpy as np
from sklearn.lda import LDA
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn import svm
from sklearn.pls import PLSCanonical
#from sklearn.metrics import zero_one_loss
#from sklearn.ensemble import AdaBoostClassifier
from scipy import random
from scipy import io as sio
import scipy.signal as signal


def siLoadImages(folder):
    imglist=list()
    dirfile = os.listdir(folder)
    for ff in dirfile:
        fullff = folder + ff
        if os.path.isfile(fullff) and siBeImage(ff):
            img = cv2.imread(fullff, cv2.CV_LOAD_IMAGE_GRAYSCALE)
#            print img.shape
#            cv2.imshow('resz', img)
#            cv2.waitKey(0)
            imglist.append((img, ff))
            
    return imglist


def siSmooth(img):
    kernel = [[1, 2, 1], [2, 4, 2], [1, 2, 1]]
    kernel = np.asarray(kernel, dtype = np.float32)
    sumv = np.sum(kernel)
    kernel /= sumv
#    print kernel
    imgs = cv2.filter2D(img, -1, kernel)
    return imgs


def siMedFilter(img):
    return signal.medfilt2d(img)


def siBeImage(filename):
    namext = os.path.splitext(filename)
    ext = namext[1].lower()
    beimg = False
    if ext == '.png' or ext == '.bmp' or ext == '.jpg' or ext == '.tiff':
        beimg = True
    return beimg
    

def siLoadImagesResize(folder, rsz=2):
    print 'do resize:', rsz, folder
    imglist=list()
    dirfile = os.listdir(folder)
    for ff in dirfile:
        fullff = folder + ff
        
        beimg = siBeImage(ff)
        if beimg and os.path.isfile(fullff):
#            print fullff
            img = cv2.imread(fullff, cv2.CV_LOAD_IMAGE_GRAYSCALE)
#            img2 = siSmooth(img)
            img2 = img
#            img2 = siMedFilter(img)
            img3 = cv2.resize(img2, (img.shape[1]/rsz, img.shape[0]/rsz)) #do resize
            print ff, img3.shape
#            allimg = np.append(img, img2, 1)
#            cv2.imshow('resz', allimg)
#            cv2.waitKey(0)
            imglist.append((img3, ff))
            
    return imglist


def siSobelY_S255(imggray):
    sbly = cv2.Sobel(imggray, cv2.CV_16S, 0, 1)
    
    sbly = sbly / 4
    sbly[sbly>255] = 255
    sbly[sbly<-255] = -255
    
    return sbly.astype(np.int16)
    

def siSobelX_S255(imggray):
    sblx = cv2.Sobel(imggray, cv2.CV_16S, 1, 0)
    
    sblx = sblx / 4
    sblx[sblx>255] = 255
    sblx[sblx<-255] = -255
    
    return sblx.astype(np.int16)


def siSobelY_U8(imggray):
    sbly = cv2.Sobel(imggray, cv2.CV_16S, 0, 1)
    
    sblxy = np.abs(sbly)
    sblxy = sblxy / 4
    sblxy[sblxy>255] = 255
    
    return sblxy.astype(np.uint8)
    

def siSobelX_U8(imggray):
    sblx = cv2.Sobel(imggray, cv2.CV_16S, 1, 0)
    
    sblxy = np.abs(sblx)
    sblxy = sblxy / 2
    sblxy[sblxy>255] = 255
    
    return sblxy.astype(np.uint8)

def siSobel_U8(imggray):
    sblx = cv2.Sobel(imggray, cv2.CV_16S, 1, 0)
    sbly = cv2.Sobel(imggray, cv2.CV_16S, 0, 1)
    
    sblxy = np.abs(sblx) + np.abs(sbly)
    sblxy = sblxy / 4
    sblxy[sblxy>255] = 255
    
    return sblxy.astype(np.uint8)


def siHistMinMax(hist, percent):
    hv = np.asarray(hist)
    allv = np.sum(hv)
    hl = hv.shape[0]
    minv = 0
    minw = 0
    for i in xrange(hl):
        minv += hv[i]*i
        minw += hv[i]
        if minw * percent > allv:
            break
    val1 = minv / minw
    
    minv = 0
    minw = 0
    for i in xrange(hl-1, -1, -1):
        minv += hv[i]*i
        minw += hv[i]
        if minw * percent > allv:
            break
    val2 = minv / minw
    if val2 < val1:
        val2 = val1
        
    return (val1, val2)


def siGetAllHists256(imggray, wndsize, step=8):
#    print imggray.shape
    imgh, imgw = imggray.shape
    ww,wh = wndsize
    bins_v = xrange(257) #not 256, because the value is the edge of bin in histogram
    histlist = list()
    for ri in xrange(0, imgh-wh, step):
        for ci in xrange(0, imgw-ww, step):
#            print (ri, ci, imgw-ww-step)
            patch = imggray[ri:ri+wh, ci:ci+ww]
#            print patch
            hist = np.histogram(patch, bins=bins_v)
#            print hist
#            print hist[0]
#            cv2.imshow('tmp', patch)
#            cv2.waitKey(0)
            histlist.append(hist[0])
#            break
#        break
    return histlist


def siGetBlockHistsN(imgblock, blocksz, binlist):
    """
    Get all block of Hists in the image
    
    Parameters
    ----------
    imgblock : array_like
        block of gray image data
    blocksz : tuple_like
        suwindow array of block, (2,2) means 2 cols and 2 rows
    binlist : list_like
        bins edges of histogram, like [0, 2, 4, 6, 8, ...]
        
    Returns
    -------
    out : list_like
        all histograms in image
    """
    imgh, imgw = imgblock.shape
    bw, bh = blocksz
    bsw = imgw / bw
    bsh = imgh / bh
#    print bsw, bsh
    histlist = []
    for ri in xrange(bh):
        for ci in xrange(bw):
            patch = imgblock[ri*bsh:(ri+1)*bsh, ci*bsw:(ci+1)*bsw]
#            cv2.imshow('cell', patch)
#            cv2.waitKey(10)
            hist = np.histogram(patch, bins=binlist)
            histlist.append(hist[0])
    histlist = np.asarray(histlist)
    histlist = np.reshape(histlist, histlist.shape[0] * histlist.shape[1])
    
    return histlist
    

def siNormalizePatch(patch):
    minv = np.min(patch)
    maxv = np.max(patch)
    npatch = (patch - minv) * 255 / (maxv - minv)
#    allimg = np.append(patch, patch, 1)
#    cv2.imshow('npatch', allimg)
#    cv2.waitKey(0)
    return npatch


def siGetAllMultiHistsN(imggray, wndsize, blocksz, binlist, step=8):
    """
    Get all blocksz of Hists in the image
    
    Parameters
    ----------
    imggray : array_like
        gray image data
    wndsize : tuple_like
        block window size, (64,64) means 64 width and 64 height
    blocksz : tuple_like
        suwindow array of block, (2,2) means 2 cols and 2 rows
    binlist : list_like
        bins edges of histogram, like [0, 2, 4, 6, 8, ...]
    step : int_like(default is 8)
        step size
        
    Returns
    -------
    out : list_like
        all histograms in image
    """
    
#    print imggray.shape
    imgh, imgw = imggray.shape
    ww,wh = wndsize
    histlist = list()
    for ri in xrange(0, imgh-wh, step):
        for ci in xrange(0, imgw-ww, step):
#            print (ri, ci, imgw-ww-step)
            patch = imggray[ri:ri+wh, ci:ci+ww]
#            print patch
            hist = siGetBlockHistsN(patch, blocksz, binlist)
#            print hist
#            print hist[0]
#            cv2.imshow('tmp', patch)
#            cv2.waitKey(0)
            histlist.append(hist)
#            break
#        break
    return histlist


def siMeanHist(hist):
    hv = np.asarray(hist)
    hl = hv.shape[0]
    meanv = 0
    wgt = np.sum(hv)
    for i in xrange(hl):
        meanv += i * hv[i]
    meanv /= wgt
    
    return meanv


def siLDA(X, y):
    lda = LDA(n_components=2)
    X_r2 = lda.fit(X, y).transform(X)
    plt.figure()
    for c, i in zip('rgb', [0, 1]):
        plt.scatter(X_r2[y == i, 0], X_r2[y == i, 1], c=c)
    plt.title('LDA')
    plt.show()


def siPCA(X, y):
    pca = PCA(n_components=2)
    X_r = pca.fit(X).transform(X)
    for c, i in zip("rgb", [0, 1]):
        plt.scatter(X_r[y == i, 0], X_r[y == i, 1], c=c)
    plt.title('PCA')
    plt.show()


def siSVM_Train(X, y, maxiter=2000):
    posy = y[y==1]
    negy = y[y==-1]
    posnum = posy.shape[0]
    negnum = negy.shape[0]
    
#    clf = svm.SVC(kernel='linear', class_weight={negnum:posnum})
    clf = svm.SVC(kernel='linear', class_weight='auto', max_iter=maxiter, verbose=True)
    print 'svm training (maxiter=%d, posnum=%d, negnum=%d)...'%(maxiter, posnum, negnum)
    clf.fit(X, y)
    return clf


def siSVM_Wb(X, y, clf, ifsave = 0):
    svs = list()
    tags = list()
    svis = clf.support_
    for i in svis:
        svs.append(X[i, :])
        tags.append(y[i])
    
    #get W
    W = clf.coef_
    W = np.asarray(W)
    #get b
    b = 0
    for sv, tag in zip(svs, tags):
        b += tag - np.dot(W, sv)
    b /= len(tags)
    
    if ifsave:
        fn = 'Wb.txt'
        f = file(fn, 'w')
        for wi in xrange(W.shape[1]):
            if wi%10 == 0:
                f.write('\n')
            ww = W[0, wi]
            f.write('%.6ff, '%(ww))
        f.write('\n')
        f.write('%.6ff'%(b))
        f.close()
        print 'saved into:', fn
    
    return (W, b)


def siSVM_Test(X, y, clf):
    print 'predict ...'
#    yy = clf.predict(X)
    rr = clf.score(X, y)
    print 'right rate(%d):%.2f%%\n'%(X.shape[0], rr*100)
    return clf

    
def siSVM_TestWb(X, y, Wb):
    print 'predict ...'
    smpnum = len(y)
    y = np.reshape(y, (smpnum, 1))
    W = Wb[0]
    b = Wb[1]
    wxb = np.dot(X, W.T) + b
    wxb[wxb>0] = 1
    wxb[wxb<=0] = -1
    wxb = wxb.astype(np.int32)
#    print np.append(wxb, y, 1)
    err = abs(wxb - y)
#    print err
    err[err>0] = 1
    errnum = np.sum(err)
    
    rr = errnum*1.0 / smpnum
    print 'right rate(%d):%.2f%%(%d/%d)\n'%(X.shape[0], (1-rr)*100, smpnum - errnum, smpnum)


def siPLS(X, y):
    plsca = PLSCanonical(n_components=2)
    plsca.fit(X, y)
    X_r = plsca.transform(X)
    print X_r

#
#def siRealBoost(X_train, y_train):
#    print 'real boosting ...'
#    n_estimators = 200
#    # A learning rate of 1. may not be optimal for both SAMME and SAMME.R
#    learning_rate = 1.
#    ada_real = AdaBoostClassifier(learning_rate=learning_rate, n_estimators=n_estimators, algorithm="SAMME.R")
#    ada_real.fit(X_train, y_train)
#    
#    
#    fig = plt.figure()
#    ax = fig.add_subplot(111)
#    
#    ada_real_err_train = np.zeros((n_estimators,))
#    for i, y_pred in enumerate(ada_real.staged_predict(X_train)):
#        ada_real_err_train[i] = zero_one_loss(y_pred, y_train)
#    print ada_real_err_train
#    ax.plot(np.arange(n_estimators) + 1, ada_real_err_train,
#            label='Real AdaBoost Train Error',
#            color='green')
#    
#    ax.set_ylim((0.0, 1.0))
#    ax.set_xlabel('n_estimators')
#    ax.set_ylabel('error rate')
#    
#    leg = ax.legend(loc='upper right', fancybox=True)
#    leg.get_frame().set_alpha(0.7)
#    
#    plt.show()


def siPermutation(X, times = 1):
    for i in xrange(times):
        X = random.permutation(X)
    return X


def siSave2MatFile(filename, datadict):
    print 'save to ', filename, '...'
    sio.savemat(filename, datadict)


def siGenerateAngleTable_U8():
    """
    value of gradX and gradY is between -255~255
    row of table is X
    col of table if Y
    """
    angletable = np.zeros((511, 511), dtype = np.int32)
    print angletable.shape
    for gX in xrange(-255, 256):
        for gY in xrange(-255, 256):
            ang = np.arctan2(gY, gX)
            ang = ang * 180 / np.pi
            if ang < 0:
                ang = 360 + ang
            angletable[gX+255, gY+255] = ang
    return angletable


def siHogOfOneBlock(angtable255, gradx, grady, cellsize=(8, 8), quantsize=16):
    """
    get hog of only one block
    """
    imgshape = gradx.shape
    angleone = 360 / quantsize
    cellrownum = imgshape[0] / cellsize[0]
    cellcolnum = imgshape[1] / cellsize[1]
    hog = np.zeros(cellrownum*cellcolnum*quantsize, dtype=np.float32)
    cellnum = 0
    for ri in xrange(0, imgshape[0], cellsize[0]):
        for ci in xrange(0, imgshape[1], cellsize[1]):
            cellstartpos = cellnum*quantsize
            for cri in xrange(cellsize[0]):
                for cci in xrange(cellsize[1]):
                    gx, gy = gradx[ri+cri, ci+cci], grady[ri+cri, ci+cci]
                    ang = angtable255[gx, gy]
                    quantang = ang / angleone
                    gm = abs(gx)+abs(gy)
                    
                    hog[cellstartpos + quantang] += gm
            cellnum += 1
    #normalize
    hognorm = np.linalg.norm(hog)
    hog /= hognorm
    
    return hog
    
        
def siShowVideoY(filename, imgshape, wait=0):
    """
    imgshape=(height, width)
    """
    imgh, imgw = imgshape
    f = open(filename, 'rb')
    framenum = 0
    while True:
        img = f.read(imgw * imgh)
        if img == '':
            print 'read over ...'
            break
        img = np.fromstring(img, dtype=np.uint8)
        img = np.reshape(img, imgshape)
#        print img.shape
        print 'frame_'+str(framenum)
        cv2.imshow('video', img.T)
        framenum += 1
        cv2.waitKey(wait)
    f.close()
    

def siGetOneFrameFromVideoY(filename, imgshape, frameid):
    """
    imgshape=(height, width)
    """
    imgh, imgw = imgshape
    f = open(filename, 'rb')
    f.seek(frameid * imgh * imgw)
    img = f.read(imgw * imgh)
    if img != '':
        img = np.fromstring(img, dtype=np.uint8)
        img = np.reshape(img, imgshape)
    f.close()
    return img
   

def siGetFilesPathesListByExt(folder, ext):
    files = os.listdir(folder)
    filelist = list()
    for fl in files:
        extmp = os.path.splitext(fl)
        extmp = extmp[1].lower()
        if extmp == ext:
            filelist.append(folder + fl)
    return filelist
    
    
def siPrintArray2D(fmt, aData):
    if len(aData.shape) != 2:
        print 'only support 2D array.'
        return
    h, w = aData.shape
    print 'Data:'
    for i in xrange(h):
        print '%3d:['%(i), 
        for j in xrange(w):
            print fmt%(aData[i, j]), 
        print ']'
    
    