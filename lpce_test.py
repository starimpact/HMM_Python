# -*- coding: utf-8 -*-
"""
Created on Fri Oct 17 15:34:55 2014

@author: mzhang
"""


import functions as funcs
import numpy as np
import cv2 as cv
import time
import lpfunctions as lpfuncs


def getInitailParamsFromSamples(obslist, statelist, max_state_value, max_obs_value):
    n_num = len(obslist)
    
    prior = np.zeros(max_state_value) + 1 / max_state_value
    
    trans = np.zeros((max_state_value, max_state_value), dtype=np.float32)
    for ni in xrange(n_num):
        state_path = statelist[ni]
        n_state = state_path.shape[0]
        for si in xrange(n_state-1):
            trans[state_path[si], state_path[si+1]] += 1
            
    trans = lpfuncs.row_normalize(trans)
    
    obs = np.ones((max_state_value, max_obs_value), dtype=np.float32)
    for ni in xrange(n_num):
        obs_chain = obslist[ni]
        state_path = statelist[ni]
        n_state = state_path.shape[0]
        for so, oo in zip(state_path, obs_chain):
            obs[so, oo] += 1
        
    obs = lpfuncs.row_normalize(obs)
    
    print obs.shape
    
    return prior, trans, obs


def train(lpinfo_list, hmmfile, maxiter=20, neednum=40):
    state_chains = []
    obs_chains = []
    
    max_state_value = 0
    max_obs_value = 128
    for i, lpi in enumerate(lpinfo_list):
        if i >= neednum:
            break
        maxv = np.max(lpi.charobj.state_chain)
        if maxv > max_state_value:
            max_state_value = maxv
        
        state_chains.append(lpi.charobj.state_chain)
        obs_chains.append(lpi.charobj.obs_chain)
    
    max_state_value += 1
    
    
    print 'state_type_num:', max_state_value, 'obs_type_num:', max_obs_value
    
    prior, trans, obs = getInitailParamsFromSamples(obs_chains, state_chains, max_state_value, max_obs_value)
    
    
    import nhmm
    
    hmm = nhmm.siNHMM(prior, trans, obs)
#    hmm.read(hmmfile)
    prob0 = [0, 0]
    prob0[0] = hmm.evaluate(obs_chains)
    
#    hmm.train(obs_chains, state_chain=state_chains, iter_max=maxiter)
    hmm.train(obs_chains, iter_max=maxiter)
    
    prob0[1] = hmm.evaluate(obs_chains)
    print prob0
    
    
    hmm.save(hmmfile)
    print 'hmm is saved into', hmmfile
    hmm.read(hmmfile)
    (prior_new, trans_new, obs_new) = hmm.getparams()
#    print prior_new
    print np.sum(prior_new)
#    print trans_new[4, :]
    print np.sum(trans_new, axis=1)
#    print obs_new
    print np.sum(obs_new, axis=1)
    
#    print obs_new[1, :]



def train_fix_width(lpinfo_list, hmmfile, maxiter=20, neednum=40):
    state_chains = []
    obs_chains = []
    
    max_state_value = 12 #fixed width of char
#    widthhist = np.zeros(128, dtype=np.float32)
    max_obs_value = 128
    for i, lpi in enumerate(lpinfo_list):
        if i >= neednum:
            break
        state_chain = lpi.charobj.state_chain
        
        state_chain[state_chain>max_state_value] = 0
        state_chains.append(state_chain)
        obs_chains.append(lpi.charobj.obs_chain)
    
    max_state_value += 1
    
    print 'fixed width:', max_state_value-1
    
    print 'state_type_num:', max_state_value, 'obs_type_num:', max_obs_value
    
    prior, trans, obs = getInitailParamsFromSamples(obs_chains, state_chains, max_state_value, max_obs_value)
    
    
    import nhmm
    
    hmm = nhmm.siNHMM(prior, trans, obs)
#    hmm.read(hmmfile)
    prob0 = [0, 0]
    prob0[0] = hmm.evaluate(obs_chains)
    
#    hmm.train(obs_chains, state_chain=state_chains, iter_max=maxiter)
    hmm.train(obs_chains, iter_max=maxiter)
    
    prob0[1] = hmm.evaluate(obs_chains)
    print prob0
    
    
    hmm.save(hmmfile)
    print 'hmm is saved into', hmmfile
    hmm.read(hmmfile)
    (prior_new, trans_new, obs_new) = hmm.getparams()
#    print prior_new
    print np.sum(prior_new)
#    print trans_new[4, :]
    print np.sum(trans_new, axis=1)
#    print obs_new
    print np.sum(obs_new, axis=1)
    
#    print obs_new[1, :]
    

def test(lpinfo_list, hmmfile):
    import nhmm
    
    hmm = nhmm.siNHMM()
    hmm.read(hmmfile)
    (prior_new, trans_new, obs_new) = hmm.getparams()
    funcs.siPrintArray2D('%5.2f', trans_new)
    
    for li, lpi in enumerate(lpinfo_list):
        print '%d:%s'%(li, lpi.img_fn)
        onelpinfo = lpi
        obs_chain = onelpinfo.charobj.obs_chain
        print 'obs:', obs_chain
        print 'state:', onelpinfo.charobj.state_chain
        find_chain, score = hmm.viterbi_one(obs_chain)
        img = lpi.charobj.grayimg
        imgh, imgw = img.shape
        print 'score:', score
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
        


    
maxiter = 20
neednum = 200
hmmfile = 'lphmm_' + str(maxiter) + '_' + str(neednum) + '_' + '_2.bin'
folderpath = '/Users/mzhang/work/LP Data2/'

#neednum = 200
lpinfo_list = lpfuncs.getall_lps(folderpath, neednum)
lpinfo_list = lpinfo_list[:neednum]
#exit()
print
train(lpinfo_list, hmmfile, maxiter, neednum)
#train_fix_width(lpinfo_list, hmmfile, maxiter, neednum)
print
test(lpinfo_list, hmmfile)



