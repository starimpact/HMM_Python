# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 13:13:55 2014

@author: mzhang

handle HMM in log space, can't work.
sigma(log(P(o|lamda,x)))
"""


import numpy as np


class siLHMM:
    def __init__(self, prior_state, trans_mat, obs_mat):
        self.__minimum = 1e-6
        self.__prior_state = np.log(prior_state + self.__minimum) #a vector, n_state
        self.__trans_mat = np.log(trans_mat + self.__minimum) #transition probability matrix, n_state X n_state
        self.__obs_mat = np.log(obs_mat + self.__minimum) #observation probability matrix, n_state X n_obsve
        self.__n_state = prior_state.shape[0]
        self.__n_obsve = obs_mat.shape[1]
        
#        print self.__prior_state
#        print self.__trans_mat
#        print self.__obs_mat
        
    
    def __foreward_mat_one(self, one_obs_data):
        """
        create alpha matrix for  a serial data: number of state by length of one serial data
        
        one_obs_data: one serial observation data
        """
#        print 'foreward matrix...'
#        print self.__trans_mat
        data_len = one_obs_data.shape[0]
        n_state = self.__n_state
        alpha_mat = np.zeros((n_state, data_len), dtype = np.float64)
        for i in xrange(n_state):
            alpha_mat[i, 0] = self.__prior_state[i] + self.__obs_mat[i, one_obs_data[0]]
        
#        print 0, alpha_mat[:, 0]
        
        for ti in xrange(1, data_len):
            for j in xrange(n_state):
                alpha_mat[j, ti] = np.sum(alpha_mat[:, ti-1] + self.__trans_mat[:, j]) + self.__obs_mat[j, one_obs_data[ti]]
#            print ti, alpha_mat[:, ti]
        
#        print alpha_mat
        
        return alpha_mat

        
    def __backward_mat_one(self, one_obs_data):
        """
        create beta matrix for a serial data: number of state by length of one observation data
        
        one_obs_data: one serial observation data
        """
#        print 'backward matrix....'
        data_len = one_obs_data.shape[0]
        n_state = self.__n_state
        beta_mat = np.zeros((n_state, data_len), dtype = np.float64)
        beta_mat[:, -1] = 0
        for ti in xrange(data_len-2, -1, -1):
            for j in xrange(n_state):
                beta_mat[j, ti] = np.sum(beta_mat[:, ti+1] + self.__trans_mat[j, :] + self.__obs_mat[:, one_obs_data[ti+1]])
                
        return beta_mat
    
    def __train_one(self, one_obs_data, state_path=None):
        """
        train on one serial data
        state_path: state chain
        """
        n_state = self.__n_state
        obs_len = one_obs_data.shape[0]
        trans_mat = self.__trans_mat
        obs_mat = self.__obs_mat
        alpha_mat = self.__foreward_mat_one(one_obs_data)
        beta_mat = self.__backward_mat_one(one_obs_data)
        
#        print 'alpha:', alpha_mat
#        print 'beta:', beta_mat
        
        
        #calc epsilon matrix
        eps_list = []
        for ti in xrange(obs_len-1):
            epsilon = np.zeros((n_state, n_state), dtype=np.float64)
            for i in xrange(n_state):
                for j in xrange(n_state):
                    epsilon[i, j] = alpha_mat[i, ti] + trans_mat[i, j] + obs_mat[j, one_obs_data[ti+1]] + beta_mat[j, ti+1]
            
#            print '++>', epsilon
            epsilon2 = np.zeros((n_state, n_state), dtype=np.float64)
            for i in xrange(n_state):
                for j in xrange(n_state):
                    tmp = epsilon - epsilon[i, j]
#                    print '-->', tmp
                    tmp = np.sum(np.exp(tmp))
                    
                    epsilon2[i, j] = np.log(1 / tmp)
#                print
                
#            print
            eps_list.append(epsilon2)
        
        #calc gamma matrix
        gamma = alpha_mat + beta_mat
#        print 'gamma:', gamma
        for ti in xrange(obs_len):
            tmp = np.copy(gamma[:, ti])
#            print tmp
            for i in xrange(n_state):
                tmp2 = tmp - tmp[i]
                tmp2 = np.sum(np.exp(tmp2))
#                print 1/tmp2, 
                gamma[i, ti] = np.log(1 / tmp2)
#            print
#        exit()
#        print gamma
        #calc new transition matrix
        eps_sum = np.zeros((n_state, n_state), dtype=np.float64)
        gamma_sum = np.zeros(n_state, dtype=np.float64)
        for ti in xrange(obs_len-1):
            eps = eps_list[ti]
            ga = gamma[:, ti]
            eps_sum += np.exp(eps)
            gamma_sum += np.exp(ga)
        
        trans_new = np.zeros((n_state, n_state), dtype=np.float64)
        for i in xrange(n_state):
            trans_new[i, :] = np.log(eps_sum[i, :] / gamma_sum[i])
        
        #calc new prior matrix
        prior_new = gamma[:, 0] #np.sum(gamma, axis=1) / (obs_len)
#        print prior_new
        
        #calc new observation matrix
        n_obs_type = obs_mat.shape[1]
        obs_new = np.zeros_like(obs_mat)
        if state_path != None:
            for i in xrange(n_state):
                gammarow = gamma[i, :]
                for j in xrange(n_obs_type):
                    gamsel = np.copy(gammarow[state_path==i and one_obs_data==j])
                    if gamsel.shape[0] == 0:
                        gamsel = np.log(np.zeros(1, dtype=np.float64) + self.__minimum)
                    tmp1 = np.exp(gamsel - gamsel[0])
                    tmp2 = np.exp(gammarow - gamsel[0])
                    obs_new[i, j] = np.log(np.sum(tmp1) / np.sum(tmp2))
        else:
            for i in xrange(n_state):
                gammarow = gamma[i, :]
                for j in xrange(n_obs_type):
                    gamsel = np.copy(gammarow[one_obs_data==j])
                    if gamsel.shape[0] == 0:
                        gamsel = np.log(np.zeros(1, dtype=np.float64) + self.__minimum)
                    tmp1 = np.exp(gamsel - gamsel[0])
                    tmp2 = np.exp(gammarow - gamsel[0])
                    obs_new[i, j] = np.log(np.sum(tmp1) / np.sum(tmp2))
#        print prior_new, trans_new, obs_new
        #evaluation
#        prob = np.sum(alpha_mat[:, -1])
#        print obs_new[0, 2]
        
        return (prior_new, trans_new, obs_new)
        
        
    def evaluate(self, obs_data):
#        print 'evaluation...'
        n_data = len(obs_data)
        prob_all = np.zeros(n_data, dtype=np.float64)
        for i in xrange(n_data):
            alpha_mat = self.__foreward_mat_one(obs_data[i])
            prob_all[i] = np.sum(alpha_mat[:, -1])
#        prob_mean = np.sum(prob_all) / n_data
#        prob_log = np.log(prob_mean)
        prob_log = np.sum(prob_all)
        
        return prob_log
    
    
    def evaluate_each(self, obs_data):
#        print 'evaluation...'
        n_data = len(obs_data)
        prob_all = np.zeros(n_data, dtype=np.float64)
        for i in xrange(n_data):
            alpha_mat = self.__foreward_mat_one(obs_data[i])
#            print alpha_mat
            prob_all[i] = np.sum(alpha_mat[:, -1])
        prob_log = prob_all
        
        return prob_log
    
    
    def save(self, fn):
        print 'save...'
        
        
    def __train_iter(self, obs_data, state_chain = None):
        n_obs = len(obs_data)
        prior_list = []
        trans_list = []
        obs_list = []
        
        
#        probs = 0
        
        if state_chain != None:
            for i in xrange(n_obs):
                prior, trans, obs = self.__train_one(obs_data[i], state_chain[i])
#                probs += prob
                prior_list.append(prior)
                trans_list.append(trans)
                obs_list.append(obs)
        else:
            for i in xrange(n_obs):
                prior, trans, obs = self.__train_one(obs_data[i])
#                print i, ':', prior
#                print i, ':', prob
#                probs += prob
                prior_list.append(prior)
                trans_list.append(trans)
                obs_list.append(obs)
                
        
        prior_new = np.exp(prior_list[0])
        trans_new = np.exp(trans_list[0])
        obs_new = np.exp(obs_list[0])
        for i in xrange(1, n_obs):
            prior_new += np.exp(prior_list[i])
            trans_new += np.exp(trans_list[i])
            obs_new += np.exp(obs_list[i])
        
        prior_new = np.log(prior_new / n_obs + self.__minimum)
        trans_new = np.log(trans_new / n_obs + self.__minimum)
        obs_new = np.log(obs_new / n_obs + self.__minimum)
#        probs /= n_obs
#        exit()
        return (prior_new, trans_new, obs_new)
        
        
    def train(self, obs_data, state_chain = None, iter_max = 10):
        """
        obs_data: it is a list object, it's elements are numpy array objects and maybe have different size
        state_chain: it is a list object, it's elements are numpy array objects and maybe have different size
        """
#        print 'train...'
        prob_mean = self.evaluate(obs_data)
        print prob_mean
#        exit(0)
#        print self.__trans_mat
        prob_old = prob_mean
        for i in xrange(iter_max):
            prior_new, trans_new, obs_new = self.__train_iter(obs_data, state_chain)
            
#            if prob_old > prob_new or abs(prob_old - prob_new) < 0.0001:
#                print 'training is over...'
#                break
            
            np.copyto(self.__prior_state, prior_new)
            np.copyto(self.__trans_mat, trans_new)
            np.copyto(self.__obs_mat, obs_new)
            
            prob_mean = self.evaluate(obs_data)
#            print 'trans:', trans_new
#            print 'prior:', prior_new
#            print 'total probabillity:', prob_mean
#            print
            print prob_mean
            prob_old = prob_mean
#        print obs_new
        print np.exp(trans_new)
        
        
    def viterbi(self, data):
        print 'viterbi...'



def test():
    print 'test...'
    data0 = [np.asarray([9, 8, 8, 7, 7, 7, 6, 6, 6, 5, 5, 5, 4, 4, 3, 2, 1, 16, 15, 15, 15, 15, 14, 14, 14, 13, 13, 13, 12, 12, 11, 10, 9, 9, 8], dtype=np.float64)-1, 
             np.asarray([8, 8, 7, 7, 7, 6, 6, 5, 5, 5, 5, 4, 4, 3, 2, 1, 1, 16, 15, 15, 15, 15, 15, 14, 14, 14, 14, 13, 12, 11, 10, 10, 9, 9, 9], dtype=np.float64)-1,
             np.asarray([7, 6, 6, 6, 6, 6, 6, 5, 5, 5, 4, 3, 2, 1, 16, 16, 16, 15, 15, 15, 15, 14, 14, 14, 14, 14, 14, 13, 11, 10, 9, 9, 8, 8, 8, 8, 7, 7, 6, 6], dtype=np.float64)-1]
    prior0 = np.asarray([1, 0, 0], dtype=np.float64)
#    prior0 = np.asarray([0.33, 0.33, 0.33], dtype=np.float64)
    trans0 = np.asarray([[0.2, 0.47, 0.33], [0, 0.45, 0.55], [0, 0, 1.0]], dtype = np.float64)
    obs0 = np.asarray([[0.02, 0.04, 0.05, 0.00, 0.12, 0.11, 0.13, 0.00, 0.06, 0.09, 0.02, 0.11, 0.06, 0.05, 0.04, 0.08], 
                       [0.12, 0.04, 0.07, 0.06, 0.03, 0.03, 0.08, 0.02, 0.11, 0.04, 0.02, 0.06, 0.06, 0.11, 0.01, 0.12],
                       [0.05, 0.04, 0.01, 0.11, 0.02, 0.08, 0.11, 0.10, 0.09, 0.02, 0.05, 0.10, 0.06, 0.00, 0.09, 0.07]], dtype = np.float64)
    
    data1 = [np.asarray([5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 5, 4], dtype=np.float64)-1, 
             np.asarray([5, 6, 6, 6, 6, 6, 6, 6, 6, 5, 5, 4], dtype=np.float64)-1, 
             np.asarray([5, 5, 5, 6, 6, 6, 6, 6, 6, 7, 6, 4, 3], dtype=np.float64)-1]
#    prior1 = np.asarray([0.5, 0.5], dtype=np.float64)
    prior1 = np.asarray([1, 0], dtype=np.float64)
    trans1 = np.asarray([[0.03, 0.97], [0, 1.0]], dtype = np.float64)
    obs1 = np.asarray([[0.05, 0.10, 0.01, 0.06, 0.02, 0.09, 0.06, 0.02, 0.10, 0.04, 0.12, 0.11, 0.03, 0.01, 0.09, 0.11],
                       [0.08, 0.09, 0.06, 0.05, 0.09, 0.10, 0.07, 0.06, 0.12, 0.03, 0.03, 0.12, 0.03, 0.01, 0.03, 0.02]], dtype = np.float64)
    
    hmm0 = siLHMM(prior0, trans0, obs0)
#    prob0 = [0, 0]
#    prob0[0] = hmm0.evaluate(data0[1:2])
    hmm0.train(data0[2:3])
#    prob0[1] = hmm0.evaluate(data0)
#    print prob0
    
#    hmm1 = siLHMM(prior1, trans1, obs1)
#    prob1 = [0, 0]
#    prob1[0] = hmm1.evaluate(data1)
#    hmm1.train(data1[0:])
#    prob1[1] = hmm1.evaluate(data1)
#    print prob1
#    
#    start = 0
#    print 'data0_cls0:', hmm0.evaluate_each(data0[start:])
#    print 'data0_cls1:', hmm1.evaluate_each(data0[start:])
#    print 'data1_cls0:', hmm0.evaluate_each(data1[start:]) 
#    print 'data1_cls1:', hmm1.evaluate_each(data1[start:])


test()
