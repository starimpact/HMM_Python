# -*- coding: utf-8 -*-
"""
Created on Mon Oct 20 23:30:14 2014

@author: mzhang
"""
import numpy as np
import nhmm

def test():
    print 'test...'
    data0 = [np.asarray([9, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 3, 2, 1, 16, 15, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 11, 10, 9, 9, 8], dtype=np.int32)-1, 
             np.asarray([8, 8, 7, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 3, 2, 1, 1, 16, 15, 15, 15, 15, 15, 14, 14, 14, 14, 13, 12, 11, 10, 10, 9, 9, 9], dtype=np.int32)-1,
             np.asarray([7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 4, 3, 2, 1, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 13, 11, 10, 9, 9, 8, 8, 8, 8, 7, 7, 6, 6], dtype=np.int32)-1]
#    prior0 = np.asarray([1, 0, 0], dtype=np.float32)
    prior0 = np.asarray([0.33, 0.33, 0.33], dtype=np.float32)
    trans0 = np.asarray([[0.2, 0.47, 0.33], [0, 0.45, 0.55], [0, 0, 1.0]], dtype = np.float32)
    obs0 = np.asarray([[0.02, 0.04, 0.05, 0.00, 0.12, 0.11, 0.13, 0.00, 0.06, 0.09, 0.02, 0.11, 0.06, 0.05, 0.04, 0.08], 
                       [0.12, 0.04, 0.07, 0.06, 0.03, 0.03, 0.08, 0.02, 0.11, 0.04, 0.02, 0.06, 0.06, 0.11, 0.01, 0.12],
                       [0.05, 0.04, 0.01, 0.11, 0.02, 0.08, 0.11, 0.10, 0.09, 0.02, 0.05, 0.10, 0.06, 0.00, 0.09, 0.07]], dtype = np.float32)
    
    data1 = [np.asarray([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4], dtype=np.int32)-1, 
             np.asarray([5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4], dtype=np.int32)-1, 
             np.asarray([5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 6, 4, 3], dtype=np.int32)-1]
#    prior1 = np.asarray([0.5, 0.5], dtype=np.float32)
    prior1 = np.asarray([1, 0], dtype=np.float32)
    trans1 = np.asarray([[0.03, 0.97], [0, 1.0]], dtype = np.float32)
    obs1 = np.asarray([[0.05, 0.10, 0.01, 0.06, 0.02, 0.09, 0.06, 0.02, 0.10, 0.04, 0.12, 0.11, 0.03, 0.01, 0.09, 0.11],
                       [0.08, 0.09, 0.06, 0.05, 0.09, 0.10, 0.07, 0.06, 0.12, 0.03, 0.03, 0.12, 0.03, 0.01, 0.03, 0.02]], dtype = np.float32)
    
    print data0[2].shape
    
    hmm0 = nhmm.siNHMM(prior0, trans0, obs0)
    prob0 = [0, 0]
    prob0[0] = hmm0.evaluate(data0)
    hmm0.train(data0[0:], iter_max=10)
    prob0[1] = hmm0.evaluate(data0)
    print prob0
    
    (prior_new, trans_new, obs_new) = hmm0.getparams()
    print prior_new, np.sum(prior_new)
    print trans_new, np.sum(trans_new, axis=1)
    print obs_new, np.sum(obs_new, axis=1)
    
#    exit()
    print '----------'
    
    hmm1 = nhmm.siNHMM(prior1, trans1, obs1)
    prob1 = [0, 0]
    prob1[0] = hmm1.evaluate(data1)
    hmm1.train(data1[0:], iter_max=10)
    prob1[1] = hmm1.evaluate(data1)
    print prob1
    
    print '----------'
    
    start = 0
    print 'data0_cls0:', hmm0.evaluate_each(data0[start:])
    print 'data0_cls1:', hmm1.evaluate_each(data0[start:])
    print 'data1_cls0:', hmm0.evaluate_each(data1[start:]) 
    print 'data1_cls1:', hmm1.evaluate_each(data1[start:])
    
    
    states0 = hmm0.viterbi(data0)
    states1 = hmm1.viterbi(data1)
    
    print states0
    print states1

test()

#def add(v):
#    v1 = v
#    v1 = v1 + 2
#    return v1
#
#v = np.zeros(10)
#v1 = add(v)
#print v
#print v1