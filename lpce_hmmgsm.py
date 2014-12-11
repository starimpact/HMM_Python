# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:11:29 2014

@author: mzhang

the observation is vector

"""

import numpy as np
import lpfunctions as lpfuncs
import functions as funcs
import cv2 as cv
import nhmmgsm as nhg


maxiter = 100
neednum = 20
hmmfile = 'lphmm_' + str(maxiter) + '_' + str(neednum) + '_' + '_2.bin'
folderpath = '/Users/mzhang/work/LP Data2/'
samplestep = 2

def getTrainingDatas(lpinfo_list):
    datadict = dict()
    for lpi in lpinfo_list:
        gimg = lpi.charobj.grayimg
        simg = lpi.charobj.sblimg
#        simg = gimg
#        zero_num = np.sum(np.sum(simg, axis=0)==0)
##        if zero_num > 0:
##            print 'zero_num', zero_num
##            exit()
#        simgf = simg.astype(np.float32) / 255
        schain = lpi.charobj.state_chain
        
        for ci in xrange(len(schain)):
            nci = schain[ci]
            if datadict.has_key(nci) == False:
                datadict[nci] = list()
            
            datadict[nci].append(simg[::samplestep, ci])
        
#        print schain, len(schain)
#        allimg = np.append(gimg, simg, axis = 1)
#        cv.imshow('hi', allimg)
#        cv.waitKey(0)
    for key in datadict:
        tmp = np.asarray(datadict[key])
        datadict[key] = tmp.T
    
    return datadict


def getInitailParamsFromSamplesGSM(lpinfo_list):
    n_num = len(lpinfo_list)
    datadict = getTrainingDatas(lpinfo_list)
    
    state_num = 0
    datadict_new = dict()
    for key in datadict:
        tmp = datadict[key]
        snum = tmp.shape[1]
        if n_num > snum or snum < tmp.shape[0]: # filter the small size set of the state
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
#        simg = gimg
#        simgf = simg.astype(np.float32) / 255
        obs_list.append(simg)
    
    prior = np.zeros(state_num) + 1 / state_num
    
    trans = np.zeros((state_num, state_num), dtype=np.float32)
    for ni in xrange(n_num):
        state_path = statepath_list[ni]
        n_state = state_path.shape[0]
        for si in xrange(n_state-1):
            trans[state_path[si], state_path[si+1]] += 1
    
    trans = lpfuncs.row_normalize(trans)
    funcs.siPrintArray2D('%.2f ', trans)
    
    
    # get gaussian single model list
    gsm_list = []
    for key in datadict:
        gsm_one = nhg.siGSM()
        data = datadict[key]
        mean = np.mean(data, axis=1)
        sigma = np.cov(data)
        gsm_one.setParams(mean, sigma)
        gsm_list.append(gsm_one)
        print key, ':', datadict[key].shape
    
    #test
#    data0 = datadict[0]
#    for i in xrange(data0.shape[1]):
#        p = gsm_list[0].calcProbability(data0[:, i])
#        print p
    
    return prior, trans, gsm_list, obs_list



def test(lpinfo_list, hmmfile):
    import nhmmgsm as nhg
    
    hmm = nhg.siNHMMGSM()
    hmm.read(hmmfile)
    (prior_new, trans_new, obs_new) = hmm.getparams()
#    funcs.siPrintArray2D('%5.2f', trans_new)
    
    for li, lpi in enumerate(lpinfo_list):
        print '%d:%s'%(li, lpi.img_fn)
        onelpinfo = lpi
        gimg = onelpinfo.charobj.grayimg[::samplestep, :]
        simg = onelpinfo.charobj.sblimg[::samplestep, :]
#        simg = gimg
#        print 'obs:', obs_chain
#        print 'state:', onelpinfo.charobj.state_chain
        find_chain, score = hmm.viterbi_one(simg)
        img = lpi.charobj.grayimg
        imgh, imgw = img.shape
#        print 'score:', score
        print 'chain:', find_chain
        cimg = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        for i in xrange(1, find_chain.shape[0]):
            if find_chain[i] == 1 or (find_chain[i] == 0 and find_chain[i-1] > 0) or (find_chain[i] > 0 and find_chain[i-1] == 0):
                cv.line(cimg, (i, 0), (i, imgh-1), (0, 0, 255), 1)
        print '------------------------------'
        allimg = np.append(lpi.charobj.grayimg, lpi.charobj.sblimg, axis=1)
        cv.imshow('allimg', allimg)
        cv.imshow('mark', cimg)
        cv.waitKey(0)

def train(lpinfo_list, hmmfile):
    prior, trans, gsm_list, obs_list = getInitailParamsFromSamplesGSM(lpinfo_list)
    nhg0 = nhg.siNHMMGSM(prior, trans, gsm_list)
#    nhg0.train(obs_list, maxiter)
    nhg0.save(hmmfile)
    print 'hmm is saved into', hmmfile
    
lpinfo_list = lpfuncs.getall_lps(folderpath, neednum)
train(lpinfo_list, hmmfile)
test(lpinfo_list, hmmfile)



