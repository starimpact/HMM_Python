# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:59:59 2014

@author: mzhang
"""


import numpy as np
import lpfunctions as lpfuncs
import cv2 as cv
import log_nhmmgmm as lognhgmm


maxiter = 10
neednum = 1100
gmmsize = 8
hmmgmmfile = 'lp_loghmmgmm(20141118)_' + str(maxiter) + '_' + str(neednum) + '_' + '_2.bin'
folderpath = '/Users/mzhang/work/LP Data2/'
samplestep = 1
neednum = 100
trainnum = neednum
teststart = neednum - 100


def getTrainingDatas(lpinfo_list):
    datadict = dict()
    for lpi in lpinfo_list:
        gimg = lpi.charobj.grayimg
        simg = lpi.charobj.sblimg
        schain = lpi.charobj.state_chain
        for ci in xrange(len(schain)):
            nci = schain[ci]
            if datadict.has_key(nci) == False:
                datadict[nci] = list()
            
            datadict[nci].append(simg[::samplestep, ci])
            
    for key in datadict:
        tmp = np.asarray(datadict[key])
        datadict[key] = tmp.T
    
    return datadict


def get_mean_cov(datavec):
    mean = np.mean(datavec, axis=1)
    cov = np.cov(datavec)
    
    return mean, cov


#def print_info_0(datavec):
#    import pylab as pl
#    mean0, cov0 = get_mean_cov(datavec)
#    histbins = xrange(0, 256)
#    hist0, histmark0 = np.histogram(datavec, bins=histbins)
#    print '--------------------'
#    print mean0, var0
#    print hist0
#    pl.hist(datavec, bins=histbins)
#    pl.show()
    
    
def initGMMParams(data, gmnum=1, maxiter=50):
    datalen = data.shape[1]
    partlen = datalen / gmnum
    gsmlist = []
    wgtlist = []
    initwgt = 1.0 / gmnum
    for i in xrange(gmnum):
        partdata = data[:, i * partlen:(i + 1) * partlen]
        mean, cov = get_mean_cov(partdata)
        gsm = lognhgmm.siLogGSM(mean, cov)
        gsmlist.append(gsm)
        wgtlist.append(initwgt)
#        print 'num:', data.shape[0], 'mean:', mean, 'sigma:', cov
    gmm = lognhgmm.siLogGMM(gsmlist, wgtlist)
    
    gmm.train(data, maxiter)
    
#    print_info_0(data)
#    exit()
    return gmm


def getInitailParamsFromSamplesGMM(lpinfo_list, gmmsize=1):
    
    print 'get init value of parameters...'
    n_num = len(lpinfo_list)
    datadict = getTrainingDatas(lpinfo_list)
    
    state_num = 0
    datadict_new = dict()
    for key in datadict:
        tmp = datadict[key]
        snum = tmp.shape[1]
        if n_num * 2 > snum: # filter the small size set of the state
            break
        state_num += 1
        datadict_new[key] = tmp
    
    datadict = datadict_new
    
    statepath_list = []
    obs_list = []
    for i, lpi in enumerate(lpinfo_list):
        state_chain = lpi.charobj.state_chain
        
        state_chain[state_chain>=state_num] = 0
        statepath_list.append(state_chain)
        
        gimg = lpi.charobj.grayimg[::samplestep, :]
        simg = lpi.charobj.sblimg[::samplestep, :]
        obs_list.append(simg)
    
    prior = np.zeros(state_num) + 1.0 / state_num
    
    trans = np.zeros((state_num, state_num), dtype=np.float32)
    for ni in xrange(n_num):
        state_path = statepath_list[ni]
        n_state = state_path.shape[0]
        for si in xrange(n_state-1):
            trans[state_path[si], state_path[si+1]] += 1
    
    trans = lpfuncs.row_normalize(trans)
    
    
    # get gaussian single model list
    gmm_maxiter = 100
    gmm_list = []
    for key in datadict:
        data = datadict[key]
        print 'proceeding state:', key, '...', '(iter:%d, size:%dx%d)'%(gmm_maxiter, data.shape[0], data.shape[1])
        gmm = initGMMParams(data, gmmsize, gmm_maxiter)
        gmm.printWeights()
        print
        gmm_list.append(gmm)
    
    return prior, trans, gmm_list, obs_list


def test(lpinfo_list, hmmfile):
    print 'data size:', len(lpinfo_list)
    hmm = lognhgmm.siLogNHMMGMM()
    hmm.read(hmmfile)
    
    for li, lpi in enumerate(lpinfo_list):
        print '%d:%s'%(li, lpi.img_fn)
        onelpinfo = lpi
        gimg = onelpinfo.charobj.grayimg[::samplestep, :]
        simg = onelpinfo.charobj.sblimg[::samplestep, :]
#        print 'obs:', obs_chain
#        print 'state:', onelpinfo.charobj.state_chain
        find_chain, score = hmm.viterbi_one(simg)
        img = lpi.charobj.grayimg
        imgh, imgw = img.shape
#        print 'score:', score
        print find_chain
        cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for i in xrange(1, find_chain.shape[0]):
            if find_chain[i] == 1 or (find_chain[i] == 0 and find_chain[i-1] > 0) or (find_chain[i] > 0 and find_chain[i-1] == 0):
                cv.line(cimg, (i, 0), (i, imgh-1), (0, 0, 255), 1)
        print '------------------------------'
        allimg = np.append(lpi.charobj.grayimg, lpi.charobj.sblimg, axis=1)
        cv.imshow('allimg', allimg)
        cv.imshow('mark', cimg)
        cv.waitKey(0)
        

def train(lpinfo_list):
    print 'data size:', len(lpinfo_list)
    prior, trans, gmm_list, obs_list = getInitailParamsFromSamplesGMM(lpinfo_list, gmmsize)
    nhg0 = lognhgmm.siLogNHMMGMM(prior, trans, gmm_list)
#    nhg0.train(obs_list, maxiter)
    nhg0.save(hmmgmmfile)
    print 'hmmgmm is saved into', hmmgmmfile

lpinfo_list = lpfuncs.getall_lps(folderpath, neednum)
#train(lpinfo_list[:trainnum])

test(lpinfo_list[teststart:], hmmgmmfile)



