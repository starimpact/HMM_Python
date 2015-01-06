# -*- coding: utf-8 -*-
"""
Created on Mon Nov 17 09:57:01 2014

@author: mzhang
"""


import numpy as np
import lpfunctions as lpfuncs
import cv2 as cv
import nhmmgmm1dim as nhgmm1d
import simpleCNN as scnn


maxiter = 100
neednum = 20
hmmgmmfile = 'lphmmgmm_' + str(maxiter) + '_' + str(neednum) + '_' + '_2.bin'
folderpath = '/Users/mzhang/work/LP Data2/'


def getTrainingDatas(lpinfo_list):
    datadict = dict()
    for lpi in lpinfo_list:
        obschain = lpi.charobj.obs_chain
        schain = lpi.charobj.state_chain
        for ci in xrange(len(schain)):
            nci = schain[ci]
            if datadict.has_key(nci) == False:
                datadict[nci] = list()
            
            datadict[nci].append(obschain[ci])
            
    for key in datadict:
        tmp = np.asarray(datadict[key])
        datadict[key] = tmp
    
    return datadict


def get_mean_var(datavec):
    mean = np.mean(datavec)
    var = np.var(datavec)
    
    return mean, var


def print_info_0(datavec):
    import pylab as pl
    mean0, var0 = get_mean_var(datavec)
    histbins = xrange(0, 256)
    hist0, histmark0 = np.histogram(datavec, bins=histbins)
    print '--------------------'
    print mean0, var0
    print hist0
    pl.hist(datavec, bins=histbins)
    pl.show()
    
    
def initGMM1DParams(data, gmnum=1):
    datalen = len(data)
    partlen = datalen / gmnum
    gsmlist = []
    wgtlist = []
    initwgt = 1.0 / gmnum
    for i in xrange(gmnum):
        partdata = data[i * partlen:(i + 1) * partlen]
        mean, var = get_mean_var(partdata)
        gsm = nhgmm1d.siGSM1D(mean, var)
        gsmlist.append(gsm)
        wgtlist.append(initwgt)
#        print 'num:', data.shape[0], 'mean:', mean, 'sigma:', var
    gmm = nhgmm1d.siGMM1D(gsmlist, wgtlist)
    
    gmm.train(data, maxiter=50)
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
        snum = tmp.shape[0]
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
        obschain = lpi.charobj.obs_chain
        obs_list.append(obschain)
    
    prior = np.zeros(state_num) + 1 / state_num
    
    trans = np.zeros((state_num, state_num), dtype=np.float32)
    for ni in xrange(n_num):
        state_path = statepath_list[ni]
        n_state = state_path.shape[0]
        for si in xrange(n_state-1):
            trans[state_path[si], state_path[si+1]] += 1
    
    trans = lpfuncs.row_normalize(trans)
    
    
    # get gaussian single model list
    gmm_list = []
    for key in datadict:
        data = datadict[key]
        gmm = initGMM1DParams(data, gmmsize)
        gmm_list.append(gmm)
    
    return prior, trans, gmm_list, obs_list


def test(lpinfo_list, hmmfile):
    hmm = nhgmm1d.siNHMMGMM1D()
    hmm.read(hmmfile)
    
    for li, lpi in enumerate(lpinfo_list):
        print '%d:%s'%(li, lpi.img_fn)
        onelpinfo = lpi
        obs_chain = onelpinfo.charobj.obs_chain
#        print 'obs:', obs_chain
#        print 'state:', onelpinfo.charobj.state_chain
        find_chain, score = hmm.viterbi_one(obs_chain)
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
    prior, trans, gmm_list, obs_list = getInitailParamsFromSamplesGMM(lpinfo_list, gmmsize=8)
    nhg0 = nhgmm1d.siNHMMGMM1D(prior, trans, gmm_list)
#    nhg0.train(obs_list, maxiter)
    nhg0.save(hmmgmmfile)
    print 'hmmgmm is saved into', hmmgmmfile
    
lpinfo_list = lpfuncs.getall_lps(folderpath, neednum)

train(lpinfo_list)

test(lpinfo_list, hmmgmmfile)



