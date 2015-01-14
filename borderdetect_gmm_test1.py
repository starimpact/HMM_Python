# -*- coding: utf-8 -*-
"""
Created on Thu Jan  8 15:28:53 2015

@author: mzhang

detect borders of LP by GMM
"""


import numpy as np
import lpfunctions as lpfuncs
import functions as funcs
import cv2 as cv
import log_nhmmgmm as lognhgmm
import cPickle
import weightCNN as wgtcnn
import time
from scipy.cluster.vq import kmeans


maxiter = 10
gmm_maxiter = 100
neednum = 400
hmmgmmfile = 'borderdetect(2015.01.09.01)_' + str(neednum) + '_.bin'
folderpath = '/Users/mzhang/work/LPR_TrainData/new/'
neednum2 = 0
folderpath2 = '/Users/mzhang/work/LPR_TrainData/old/'
stdshape1 = (28, 14) #(28, 32) #(28, 14)

#neednum = 100
trainnum = neednum + neednum2 # - 4000
teststart = 0 #neednum - 4000


gmmsize = 4
gwndlen = 32
gmmvecdim = 2
histstep = 1
bTrain = False

def getNormalizedHist(colpart):
    sbxhist = np.sum(colpart, axis=1)
    minv = np.min(sbxhist)
    maxv = np.max(sbxhist)
    nhist = (sbxhist - minv) * 255 / (maxv - minv + 1)
    
    return nhist


def sumHist(hist, v):
    hlen = len(hist)
    rnum = hlen / v
    tmp = np.reshape(hist, (rnum, v))
#    print tmp
    newhist = np.sum(tmp, axis=1)
    newhist /= v
    
    return newhist
    


def get_mean_cov(datavec):
    mean = np.mean(datavec, axis=1)
#    print mean
    cov = np.cov(datavec)
    
    return mean, cov


def calcGMMInitValue(data, k):
    print 'calc gmm init value...'
    dnum = data.shape[0]
#    idx = np.random.permutation(range(dnum))
#    means = data[idx[:k], :]
    
    print 'guess means of gm by kmeans...'
    kret = kmeans(data, k, iter=10)
#    print kret
    means = kret[0]
#    dists = kret[1]
#    print means
#    print dists
    print 'kmeans is over....'
    
    distmat = np.zeros((dnum, k))
    for ki in xrange(k):
        tmp = data - means[ki, :]
        tmp = tmp * tmp
        tmp = np.sum(tmp, axis=1)
        distmat[:, ki] = np.sqrt(tmp)
    
    
    nums = np.zeros(k)
    meanlist = []
    covlist = []
    wgtlist = []
    for ki in xrange(k):
        minidx = np.argmin(distmat, axis=1)
        nums[ki] = np.sum(minidx==ki)
        datatmp = data[minidx==ki, :]
        meanv, covv = get_mean_cov(datatmp.T)
        meanlist.append(meanv)
        covlist.append(covv)
        wgtlist.append(nums[ki] * 1.0 / dnum)
    
    return meanlist, covlist, wgtlist


def getTrainingDatas(lpinfo_list):
    print 'get training data....'
    datadict = dict()
    key1 = 1
    key2 = 2
    datadict[key1] = list()
    datadict[key2] = list()
    halfw = gmmvecdim / 2
    wndlen = gwndlen
    mean1 = 0
    mean2 = 0
    for lpi in lpinfo_list:
        gimg = cv.imread(lpi.img_fn, cv.CV_LOAD_IMAGE_GRAYSCALE)
#        chbbs = lpi.char_bndbox
        bbs = lpi.lp_bndbox
        margin = (bbs[3]-bbs[1]) / 2
        marginw = (bbs[2]-bbs[0]) / 2
        bbs2 = [np.max([0, bbs[0]-marginw]), np.max([0, bbs[1]-margin]), 
                np.min([gimg.shape[1]-1, bbs[2]+marginw]), np.min([gimg.shape[0]-1, bbs[3]+margin])]
        lpimg = np.copy(gimg[bbs2[1]:bbs2[3]+1, bbs2[0]:bbs2[2]+1])
        newbbs = [bbs[0]-bbs2[0], bbs[1]-bbs2[1], bbs[2]-bbs2[0], bbs[3]-bbs2[1]]
        sblx = funcs.siSobelX_U8(lpimg)
        if 0:
            tmpgimg = sblx
            clpimg = np.zeros((lpimg.shape[0], lpimg.shape[1], 3), dtype=np.uint8)
            clpimg[:, :, 0] = tmpgimg
            clpimg[:, :, 1] = tmpgimg
            clpimg[:, :, 2] = tmpgimg
            cv.rectangle(clpimg, (newbbs[0], newbbs[1]), (newbbs[2], newbbs[3]), [0, 255, 0])
            cv.imshow('lpimg', clpimg)
            cv.waitKey(0)

        ystart = newbbs[1]
        yend = newbbs[3]
        if ystart <= halfw or yend+halfw >= lpimg.shape[0]:
            continue
        
        for ci in xrange(newbbs[0], newbbs[2]-wndlen, 4):
            colpart = sblx[:, ci:ci+wndlen]
            nhist = getNormalizedHist(colpart)
#            print nhist
            
            hist1 = nhist[ystart-halfw:ystart+halfw]
            hist2 = nhist[yend-halfw:yend+halfw]
#            print hist1, '-', hist2
            if 0 and hist1[0] > 200:
                print hist1
            
            hist1 = sumHist(hist1, histstep)
            hist2 = sumHist(hist2, histstep)
            datadict[key1].append(hist1)
            datadict[key2].append(hist2)
#            print hist2
            
            if 0 and hist1[0] > 200:
                colpart2 = lpimg[:, ci:ci+wndlen]
                allimg = np.append(colpart2, colpart, axis=1)
                cv.imshow('allimg', allimg)
                cv.waitKey(0)
#            print hist2
            mean1 += hist1
            mean2 += hist2
    snum = len(datadict[key1])
    print mean1 / snum
    snum = len(datadict[key2])
    print mean2 / snum
    for key in datadict:
        tmp = np.asarray(datadict[key])
        datadict[key] = tmp.T
#    exit()
    return datadict
    
    
def initGMMParams(data, gmnum=1, maxiter=50, trainratio=1.0):
    datalen = data.shape[1]
    
    #get half samples to train.
    print 'get', trainratio, 'data to train:', datalen, '->',
    dataT = data.T
    dataT = np.random.permutation(dataT)
    neednum = np.int(datalen * trainratio)
    subdataT = dataT[:neednum, :]
    subdata = subdataT.T
    datalen = subdata.shape[1]
    print datalen
    
    meanlist, covlist, wgtlist = calcGMMInitValue(dataT[:datalen/4, :], gmnum)
    
    gsmlist = []
    for i in xrange(gmnum):
        mean, cov = meanlist[i], covlist[i]
        gsm = lognhgmm.siLogGSM(mean, cov)
        gsmlist.append(gsm)
#        print 'num:', data.shape[0], 'mean:', mean, 'sigma:', cov
    gmm = lognhgmm.siLogGMM(gsmlist, wgtlist)
    
    gmm.train(subdata, maxiter)
    
#    print_info_0(data)
#    exit()
    return gmm


def getInitailParamsFromSamplesGMM(lpinfo_list, gmmsize=1):
    
    n_num = len(lpinfo_list)
    datadict = getTrainingDatas(lpinfo_list)
    statenum = len(datadict)
    state_num = 0
    datadict_new = dict()
    for key in datadict:
        tmp = datadict[key]
        snum = tmp.shape[1]
        dimnum = tmp.shape[0]
        print 'state:%s samplenum:%d/%d'%(key, snum, n_num)
        if dimnum * 40 > snum: # filter the small size set of the state
            break
        state_num += 1
        datadict_new[key] = tmp
    
    datadict = datadict_new
    
    # get gaussian single model list
    gmm_list = []
    for key in datadict:
        data = datadict[key]
        print 'proceeding state:%d/%d'%(key, statenum), '...', '(iter:%d, size:%dx%d)'%(gmm_maxiter, data.shape[0], data.shape[1])
        trainratio = 0.9
        gmm = initGMMParams(data, gmmsize, gmm_maxiter, trainratio=trainratio)
        gmm.printWeights()
        print
        gmm_list.append(gmm)
    
    return gmm_list


def train(lpinfo_list, hmmgmmfile):
    print 'training gmm...'
    
    print 'data size:', len(lpinfo_list)
    gmm_list = getInitailParamsFromSamplesGMM(lpinfo_list, gmmsize)
    lnhg = lognhgmm.siLogNHMMGMM(None, None, gmm_list)
    lnhg.save(hmmgmmfile)
    lnhg.saveTXT(hmmgmmfile+'.txt')


def findLinesByDP(graph):
#    print graph.shape
    imgh, imgw = graph.shape
    pathgraph = np.zeros_like(graph, dtype=np.int32)
    maxgraph = np.copy(graph)
    for ri in xrange(1, imgh):
        rowmaxpre = maxgraph[ri-1, :]
        rowmaxnow = maxgraph[ri, :]
        pathrownow = pathgraph[ri, :]
        for ci in xrange(1, imgw-1):
            rowseg = rowmaxpre[ci-1:ci+2]
            pathrownow[ci] = np.argmax(rowseg) - 1
            rowmaxnow[ci] += np.max(rowseg)
    xx = []
    lastrow = imgh - 1
    rowmaxlast = maxgraph[lastrow, :]
    rowmaxpathi = np.argmax(rowmaxlast)
#    print rowmaxlast
    maxpathpre_subidx = pathgraph[lastrow, rowmaxpathi]
    xxone = [lastrow, rowmaxpathi]
    xx.append(np.copy(xxone))
    while(1):
        xxone[0] -= 1
        xxone[1] += maxpathpre_subidx
        if graph[xxone[0], xxone[1]] > 0:
            xx.append(np.copy(xxone))
        if xxone[0] == 0 or xxone[1] == 0:
            break
        maxpathpre_subidx = pathgraph[xxone[0], xxone[1]]
#        print xxone[0], maxpathi
    xx = np.asarray(xx)
    return xx
    

def test(lpinfo_list, hmmgmmfile):
    lnhg = lognhgmm.siLogNHMMGMM()
    lnhg.read(hmmgmmfile)
    state, trans, gmmlist = lnhg.getparams()
    halfw = gmmvecdim / 2
    wndlen = gwndlen
    for pnumi, lpi in enumerate(lpinfo_list):
        print pnumi
        if pnumi < 4:
            continue
        gimg = cv.imread(lpi.img_fn, cv.CV_LOAD_IMAGE_GRAYSCALE)
#        chbbs = lpi.char_bndbox
        bbs = lpi.lp_bndbox
        margin = (bbs[3]-bbs[1])
        marginwl = (bbs[2]-bbs[0]) / 4
        marginwr = (bbs[2]-bbs[0]) / 4
        bbs2 = [np.max([0, bbs[0]-marginwl]), np.max([0, bbs[1]-margin]), 
                np.min([gimg.shape[1]-1, bbs[2]+marginwr]), np.min([gimg.shape[0]-1, bbs[3]+margin])]
        lpimg = np.copy(gimg[bbs2[1]:bbs2[3]+1, bbs2[0]:bbs2[2]+1])
        newbbs = [bbs[0]-bbs2[0], bbs[1]-bbs2[1], bbs[2]-bbs2[0], bbs[3]-bbs2[1]]
        sblx = funcs.siSobelX_U8(lpimg)
        
        cv.imwrite('img/'+str(pnumi)+'.bmp', lpimg.T)
        
        imgh, imgw = lpimg.shape
        scoremat = np.zeros_like(lpimg, dtype=np.float32)
        for ci in xrange(0, imgw-wndlen, 1):
            colpart = sblx[:, ci:ci+wndlen]
            nhist = getNormalizedHist(colpart)
#            print nhist
            for ri in xrange(halfw, len(nhist)-halfw):
                hist = nhist[ri-halfw:ri+halfw]
                hist = sumHist(hist, histstep)
                probval = gmmlist[0].calcLogProbability(hist)
                scoremat[ri, ci] = probval
#                print '%.2f,'%(probval),
#                if ri==newbbs[1] or ri==newbbs[3]:
#                    print '-----'
#                else:
#                    print
#            break
        subscoremat = scoremat[halfw:imgh-halfw, :imgw-wndlen]
        minv = np.min(subscoremat)
        maxv = np.max(subscoremat)
        normscoremat = (subscoremat - minv) * 255 / (maxv - minv)
        scoremat[halfw:imgh-halfw, :imgw-wndlen] = normscoremat
        xx = findLinesByDP(scoremat.T)
        normscoremat = normscoremat.astype(np.uint8)
        mark = np.zeros_like(lpimg)
        mark[halfw:imgh-halfw, :imgw-wndlen] = normscoremat
        mark[mark<200] = 0
        if 1:
#            tmpgimg = sblx
            tmpgimg = lpimg
            clpimg = cv.cvtColor(tmpgimg, cv.COLOR_GRAY2BGR)
            cmark = cv.cvtColor(mark, cv.COLOR_GRAY2BGR)
            red = np.asarray([0, 0, 255], dtype=np.uint8)
            
            cv.rectangle(clpimg, (newbbs[0], newbbs[1]), (newbbs[2], newbbs[3]), [0, 255, 0])
            cv.rectangle(cmark, (newbbs[0], newbbs[1]), (newbbs[2], newbbs[3]), [0, 255, 0])
            
            cmark = clpimg / 2 + cmark / 2
#            for ci in xrange(imgw):
#                col = mark[:, ci]
#                for ri in xrange(1, imgh-1):
#                    if col[ri] > col[ri-1] and col[ri] > col[ri+1]:
#                        cmark[ri, ci, :] = red
#            print xx.shape
            for pi in xrange(xx.shape[0]):
                pnt = xx[pi, :]
                clpimg[pnt[1], pnt[0], :] = red
            
            allimg = np.append(clpimg, cmark, axis=0)
            cv.imshow('lpimg', allimg)
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
    
    train(lpinfo_list, hmmgmmfile)
else:
#    folderpath = '/Users/mzhang/work/LPR_TrainData/slope/'
    lpinfo_list = lpfuncs.getall_lps3(folderpath, neednum)
    test(lpinfo_list, hmmgmmfile)


