# -*- coding: utf-8 -*-
"""
Created on Sat Nov 22 09:31:17 2014

@author: mzhang
"""


import numpy as np
import lpfunctions as lpfuncs
import functions as funcs
import cv2 as cv
import log_nhmmgmm as lognhgmm
import cPickle
import weightCNN as wgtcnn
import time

maxiter = 10
gmm_maxiter = 100
neednum = 400
gmmsize = 4
hmmgmmfile = 'lp_loghmmgmm(2014129.2)_hmmmaxiter[' + str(maxiter) + ']_samplenum[' + str(neednum) + ']_gmmsize[' + str(gmmsize) + ']_gmmmaxiter[' + str(gmm_maxiter) + ']' + '.bin'
#folderpath = '/Users/mzhang/work/LP Data2/'
folderpath = '/Users/mzhang/work/LPR_TrainData/new/'
neednum2 = 0
folderpath2 = '/Users/mzhang/work/LPR_TrainData/old/'
samplestep = 1
stdshape = (28, 14) #(28, 32) #(28, 14)
gmmvecdim = 8
#neednum = 100
trainnum = neednum + neednum2 # - 4000
teststart = 0 #neednum - 4000


def getTrainingDatas(lpinfo_list):
    datadict = dict()
    halfw = gmmvecdim / 2
    for lpi in lpinfo_list:
        obs_chain = lpi.charobj.obs_chain
        schain = lpi.charobj.state_chain
        for ci in xrange(halfw, len(schain)-halfw):
            nci = schain[ci]
            if datadict.has_key(nci) == False:
                datadict[nci] = list()
            
            datadict[nci].append(obs_chain[ci-halfw:ci+halfw])
    
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
    
    partlen = datalen / gmnum
    gsmlist = []
    wgtlist = []
    initwgt = 1.0 / gmnum
    for i in xrange(gmnum):
        partdata = subdata[:, i * partlen:(i + 1) * partlen]
        mean, cov = get_mean_cov(partdata)
        gsm = lognhgmm.siLogGSM(mean, cov)
        gsmlist.append(gsm)
        wgtlist.append(initwgt)
#        print 'num:', data.shape[0], 'mean:', mean, 'sigma:', cov
    gmm = lognhgmm.siLogGMM(gsmlist, wgtlist)
    
    gmm.train(subdata, maxiter)
    
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
        dimnum = tmp.shape[0]
        print 'state:%s samplenum:%d/%d'%(key, snum, n_num)
        if dimnum * 40 > snum: # filter the small size set of the state
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
        simg = gimg
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
    gmm_list = []
    for key in datadict:
        data = datadict[key]
        print 'proceeding state:', key, '...', '(iter:%d, size:%dx%d)'%(gmm_maxiter, data.shape[0], data.shape[1])
        trainratio = 0.75
        if key==0:
            trainratio = 0.25
        gmm = initGMMParams(data, gmmsize, gmm_maxiter, trainratio=trainratio)
        gmm.printWeights()
        print
        gmm_list.append(gmm)
    
    return prior, trans, gmm_list, obs_list


def doReadHMM_AdjustStateNum(hmmfile, statenum=-1):
    """
    satenum: -1 means no change
    """
    
    hmm = lognhgmm.siLogNHMMGMM()
    hmm.read(hmmfile)
    
    
    if statenum < 0:
        return hmm
    
    print 'fixed width:', statenum-1
    
    prior_state, trans_mat, gmm_list = hmm.getparams()
    n_state = len(prior_state)
    print 'total state number:', n_state
    
    if n_state < statenum:
        print 'error! only has %d states, but you need %d states.'%(n_state, statenum)
        return hmm
    
    prior_state = np.zeros(statenum)
    prior_state[0] = 1.0
    prior_state[1:] = (1-prior_state[0]) / (statenum-1)
    trans_mat = np.zeros((statenum, statenum))
    for i in xrange(statenum-1):
        trans_mat[i, i+1] = 1.0
    trans_mat[0, 0] = 0.5
    trans_mat[0, 1] = 0.5
    trans_mat[statenum-1, 0] = 0.5
    trans_mat[statenum-1, 1] = 0.5
#        for i in xrange(n_state - statenum):
#            print statenum, n_state, i
#            gmm_list.pop(-1)
    hmm = lognhgmm.siLogNHMMGMM(prior_state, trans_mat, gmm_list[:statenum])
    
    prior_state, trans_mat, gmm_list = hmm.getparams()
    n_state = len(prior_state)
    print n_state
    print prior_state
    funcs.siPrintArray2D('%.2f, ', trans_mat)
    
    return hmm


def getRectsFromChain(chain):
    rectlist = []
    s = 0
    e = 0
    for i in xrange(len(chain)-1):
        if chain[i] == 1:
            s = i
        elif s > 0 and (chain[i+1] == 1 or chain[i+1] == 0):
            e = i
            rectlist.append((s, e))
            s = 0
            e = 0
    
    return rectlist


def checkMatch(findchain, orichain):
    bMatch = False
    
    if len(findchain) != len(orichain):
        return bMatch
        print 'checkMatch error!!'
    
    findrects = getRectsFromChain(findchain)
    orirects = getRectsFromChain(orichain)
    samenum = 0
    for orect in orirects:
        for frect in findrects:
            diffs = np.abs(frect[0] - orect[0])
            diffe = np.abs(frect[1] - frect[1])
            if diffs <= 3 and diffe <= 3:
                samenum += 1
    if len(orirects) - samenum <= 0:
        bMatch = True
    
    return bMatch


def test(lpinfo_list, hmmfile, bShow=False):
    print 'test...'
    
    print 'data size:', len(lpinfo_list)
    
    hmm = lognhgmm.siLogNHMMGMM()
    hmm.read(hmmfile)
    prior_state, trans_mat, gmm_list = hmm.getparams()
    n_state_max = len(prior_state)
    print 'max number state:', n_state_max
#    widthlist = [13, 14, 15]
    widthlist = range(4, n_state_max)
    
    hmmlist = []
    for wd in widthlist:
        hmm = doReadHMM_AdjustStateNum(hmmfile, wd)
        hmmlist.append(hmm)
    
    halfw = gmmvecdim / 2
    errorlpinfolist = []
    errnum = 0
    for li, lpi in enumerate(lpinfo_list):
        print '%d:%s'%(li, lpi.img_fn)
        onelpinfo = lpi
        obs_chain = onelpinfo.charobj.obs_chain
        img = lpi.charobj.grayimg
        imgh, imgw = img.shape
        obsdata = np.zeros((2*halfw, imgw), dtype=np.float32)
        for oci in xrange(halfw, imgw-halfw):
            obsdata[:, oci] = obs_chain[oci-halfw:oci+halfw]
        
        cimgori = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
        cimgall = cimgori
        scorelist = []
        chainlist = []
        for hi, hmm in enumerate(hmmlist):
            find_chain, score = hmm.viterbi_one(obsdata)
            scorelist.append(score)
            chainlist.append(find_chain)
        
        bestcidx = np.argmax(scorelist)
        bestchain = chainlist[bestcidx]
        bMatch = checkMatch(bestchain, onelpinfo.charobj.state_chain)
        
#        print bestchain
        if bMatch == False:
            errorlpinfolist.append(lpi)
            errnum += 1
        
        print 'the best is', bestcidx, ', and the width is', widthlist[bestcidx], ', and the score is', scorelist[bestcidx], bMatch, 'errnum:', errnum
        
        cimg = np.copy(cimgori)
        for i in xrange(1, len(bestchain)):
            if bestchain[i] == 1 or (bestchain[i] == 0 and bestchain[i-1] > 0) or (bestchain[i] > 0 and bestchain[i-1] == 0):
                cv.line(cimg, (i, 0), (i, imgh-1), (255, 0, 0), 1)
        cimgall = cimg
        orichain = onelpinfo.charobj.state_chain
        cimg = np.copy(cimgori)
        for i in xrange(1, len(bestchain)):
            if orichain[i] == 1 or (orichain[i] == 0 and orichain[i-1] > 0) or (orichain[i] > 0 and orichain[i-1] == 0):
                cv.line(cimg, (i, 0), (i, imgh-1), (0, 0, 255), 1)
        cimgall = np.append(cimgall, cimg, axis=0)
        print '------------------------------'
        cv.imshow('mark', cimgall)
        if bShow:
            mask = np.zeros_like(img)
            for wi in xrange(halfw, imgw-halfw):
                mask[:, wi] = int(obs_chain[wi])

            allimg = img / 2 + mask / 2
            cv.imshow('result', allimg)
            print bestchain
            cv.waitKey(0)
        else:
            cv.waitKey(40)
    print 'errnum:%d/%d'%(errnum, len(lpinfo_list))
    
    
    if bShow == False:
        errfile = 'errorlpinfolist.bin'
        cPickle.dump(errorlpinfolist, open(errfile, 'wb'))
        print 'error file is saved into', errfile
    
    return errorlpinfolist


def testerror(hmmfile):
    errfile = 'errorlpinfolist.bin'
    errorlpinfolist = cPickle.load(open(errfile, 'rb'))
    test(errorlpinfolist, hmmfile, bShow=True)


def train(lpinfo_list, hmmgmmfile):
    print 'training hmmgmm...'
    
    print 'data size:', len(lpinfo_list)
    prior, trans, gmm_list, obs_list = getInitailParamsFromSamplesGMM(lpinfo_list, gmmsize)
    nhg0 = lognhgmm.siLogNHMMGMM(prior, trans, gmm_list)
#    nhg0.train(obs_list, maxiter)
    nhg0.save(hmmgmmfile)
    print 'hmmgmm is saved into', hmmgmmfile


if 1:
    cost_times = {}
    t1 = time.time()
    lpinfo_list2, whist2 = lpfuncs.getall_lps2(folderpath2, neednum2, stdshape[0], ifstrech=False)
    lpinfo_list, whist = lpfuncs.getall_lps2(folderpath, neednum, stdshape[0], ifstrech=False)
    for key2 in whist2:
        if whist.has_key(key2):
            whist[key2] += whist2[key2]
        else:
            whist[key2] = whist2[key2]
    print 'width list:'
    print whist
    lpinfo_list = lpinfo_list + lpinfo_list2
    t2 = time.time()
    cost_times['getall_lps2'] = t2-t1
    print 'total sample number:', len(lpinfo_list)
    
    
    t1 = time.time()
    #training char classifier
    print 'training char classifier ... '
    wgtcnn.train(lpinfo_list[:trainnum], batch_size=400, ishape=stdshape, nkerns=4, h_out=16, sampletype=1, cnnparamsfile='wgtcnn.params.char.bin', cnnparamsfile_restore=None)
    t2 = time.time()
    cost_times['wgtcnn.train_char'] = t2-t1
    
    t1 = time.time()
    #training left and right border of LP classifier
    print 'training left and right border of LP classifier ... '
    wgtcnn.train(lpinfo_list[:trainnum], batch_size=400, ishape=stdshape, nkerns=8, h_out=128, sampletype=2, cnnparamsfile='wgtcnn.params.lrborder.bin', cnnparamsfile_restore=None)
    t2 = time.time()
    cost_times['wgtcnn.train_lrborder'] = t2-t1
    
    exit()
    
    t1 = time.time()
    wgtcnn.fillObsChain(lpinfo_list, stdsize=stdshape, nkerns=4, h_out=16, sampletype=1, cnnparamsfile='wgtcnn.params.char.bin')
    t2 = time.time()
    cost_times['wgtcnn.fillObsChain'] = t2-t1
    
    t1 = time.time()
    train(lpinfo_list[:trainnum], hmmgmmfile)
    t2 = time.time()
    cost_times['hmmgmm.train'] = t2-t1
    
    t1 = time.time()
    errorlpinfolist = test(lpinfo_list[teststart:], hmmgmmfile, bShow=False)
    t2 = time.time()
    cost_times['test'] = t2-t1
    for key in cost_times:
        print key, ':', int(cost_times[key]), 's.'
    
elif 0:
#    hmmgmmfile = 'lp_loghmmgmm(20141120.1)_hmmmaxiter[10]_samplenum[1100]_gmmsize[8]_gmmmaxiter[60].bin'
    testerror(hmmgmmfile)

#save txt info
if 1:
    wgtcnn.saveCNNParam2TXT('_%dx%d_'%(stdshape[0], stdshape[1]), cnnparamsfile = 'wgtcnn.params.bin')
    exit()
    hmm = lognhgmm.siLogNHMMGMM()
    hmm.read(hmmgmmfile)
    hmm.saveTXT('log_nhmmgmm_info.txt')
    



