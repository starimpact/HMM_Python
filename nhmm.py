# -*- coding: utf-8 -*-
"""
Created on Thu Oct 16 17:09:48 2014

@author: mzhang

discrete version of hmm.

precision is float32

normalize in each layer.
this is the best and stable method.


"""


import numpy as np
import cPickle
import time
import functions as funcs

gshowtime = False

class siNHMM:
    def __init__(self, prior_state=None, trans_mat=None, obs_mat=None):
        self.__minimum = 1e-32
        
        if prior_state is not None:
            self.__prior_state = prior_state #a vector, n_state
            self.__n_state = prior_state.shape[0]
        if trans_mat is not None:
            self.__trans_mat = trans_mat #transition probability matrix, n_state X n_state
        if obs_mat is not None:
            self.__obs_mat = obs_mat #observation probability matrix, n_state X n_obsve
            self.__n_obsve = obs_mat.shape[1]
        
    
    
    def __normalize(self, vec):
#        vn = np.linalg.norm(vec)
        if np.sum(vec) < self.__minimum * vec.shape[0]:
            vec1 = vec + self.__minimum
        else:
            vec1 = vec
#        vec1 = np.maximum(vec, self.__minimum)
        
        vn = np.sum(vec1)
        nvec = vec1 / vn
        
        return vn, nvec
        
        
    def __foreward_mat_one(self, one_obs_data):
        """
        create alpha matrix for  a serial data: number of state by length of one serial data
        
        one_obs_data: one serial observation data
        """
#        print 'foreward matrix...'
#        print self.__trans_mat
#        print one_obs_data
        data_len = one_obs_data.shape[0]
        scales = np.zeros(data_len, dtype = np.float32)
        n_state = self.__n_state
        alpha_mat = np.zeros((n_state, data_len), dtype = np.float32)
        for i in xrange(n_state):
            alpha_mat[i, 0] = self.__prior_state[i] * self.__obs_mat[i, one_obs_data[0]]
#            print self.__obs_mat[i, one_obs_data[0]], self.__prior_state[i]
        scales[0], alpha_mat[:, 0] = self.__normalize(alpha_mat[:, 0])
#        print alpha_mat[:, 0]
#        exit()
        
        for ti in xrange(1, data_len):
            for j in xrange(n_state):
                alpha_mat[j, ti] = np.dot(alpha_mat[:, ti-1], self.__trans_mat[:, j]) * self.__obs_mat[j, one_obs_data[ti]]
            scales[ti], alpha_mat[:, ti] = self.__normalize(alpha_mat[:, ti])
#            print ti, alpha_mat[:, ti]
        
#        print alpha_mat
        
        return alpha_mat, scales

        
    def __backward_mat_one(self, one_obs_data):
        """
        create beta matrix for a serial data: number of state by length of one observation data
        
        one_obs_data: one serial observation data
        """
#        print 'backward matrix....'
        data_len = one_obs_data.shape[0]
        n_state = self.__n_state
        beta_mat = np.zeros((n_state, data_len), dtype = np.float32)
        beta_mat[:, -1] = 1.0
        for ti in xrange(data_len-2, -1, -1):
            for j in xrange(n_state):
                beta_mat[j, ti] = np.dot(beta_mat[:, ti+1], self.__trans_mat[j, :]*self.__obs_mat[:, one_obs_data[ti+1]])
            scale, beta_mat[:, ti] = self.__normalize(beta_mat[:, ti])
                
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
        
        if gshowtime:
            time_start = time.time()
        alpha_mat, scales = self.__foreward_mat_one(one_obs_data)
        beta_mat = self.__backward_mat_one(one_obs_data)
        if gshowtime:
            time_end = time.time()
            print 'alpha_beta_calc:%.2f ms'%((time_end - time_start) * 100)
        
#        print 'alpha:', alpha_mat
#        print 'beta:', beta_mat
        #calc epsilon matrix
        if gshowtime:
            time_start = time.time()
        eps_list = []
        for ti in xrange(obs_len-1):
            epsilon = np.zeros((n_state, n_state), dtype=np.float32)
            for i in xrange(n_state):
                for j in xrange(n_state):
                    epsilon[i, j] = alpha_mat[i, ti] * trans_mat[i, j] * obs_mat[j, one_obs_data[ti+1]] * beta_mat[j, ti+1]
            epsilon /= np.sum(epsilon)
            eps_list.append(epsilon)
        if gshowtime:
            time_end = time.time()
            print 'epsilon_calc:%.2f ms'%((time_end - time_start) * 100)
        
        #calc gamma matrix
        gamma = alpha_mat * beta_mat
#        print 'gamma:', gamma
#        print
        for ti in xrange(obs_len):
            gamma_tmp = gamma[:, ti]
#            gamma[:, ti] = gamma_tmp / np.sum(gamma_tmp)
            tmp2, gamma[:, ti] = self.__normalize(gamma_tmp)
#        print gamma
        
#        exit()
        #calc new transition matrix
        if state_path is None:
            eps_sum = np.zeros((n_state, n_state), dtype=np.float32)
            gamma_sum = np.zeros(n_state, dtype=np.float32)
            for ti in xrange(obs_len-1):
                eps = eps_list[ti]
                ga = gamma[:, ti]
                eps_sum += eps
                gamma_sum += ga + self.__minimum
            
            trans_new = np.zeros((n_state, n_state), dtype=np.float32)
            for i in xrange(n_state):
    #            print i
                trans_new[i, :] = eps_sum[i, :] / (gamma_sum[i])
                tmp1, trans_new[i, :] = self.__normalize(trans_new[i, :])
        else:
            trans_new = np.zeros((n_state, n_state), dtype=np.float32)
            for i in xrange(obs_len-1):
                trans_new[state_path[i], state_path[i+1]] += 1
            for i in xrange(n_state):
                tmp1, trans_new[i, :] = self.__normalize(trans_new[i, :])
        
        
        #calc new prior matrix
#        prior_new = np.sum(gamma, axis=1) / (obs_len)
        prior_new = gamma[:, 0]
        tmp1, prior_new = self.__normalize(prior_new)
#        print prior_new
        
        
        #calc new observation matrix
        if gshowtime:
            time_start = time.time()
        n_obs_type = obs_mat.shape[1]
        obs_new = np.zeros_like(obs_mat)
#        print state_path.shape[0], one_obs_data.shape[0]
#        print 'n_state', n_state
        if state_path is not None:
            for i in xrange(n_state):
#                gammarow = np.maximum(gamma[i, :], self.__minimum)
                gammarow = gamma[i, :] + self.__minimum
#                print gammarow.shape[0]
                gammasum = np.sum(gammarow)
                sel_state = state_path==i
                for j in xrange(n_obs_type):
                    sel_obs = one_obs_data==j
                    sel_all = np.zeros_like(sel_state)
                    for si, sel in enumerate(zip(sel_state, sel_obs)):
                        tmp = sel[0] and sel[1]
                        sel_all[si] = tmp
#                    print sel_all
                    
                    selectelem = gammarow[sel_all]
                    tmp = np.sum(selectelem)
#                    print selectelem
                    obs_new[i, j] = tmp / (gammasum)
                tmp1, obs_new[i, :] = self.__normalize(obs_new[i, :])
                
#            print np.sum(obs_new, axis=1)
        else:
            for i in xrange(n_state):
#                gammarow = np.maximum(gamma[i, :], self.__minimum)
                gammarow = gamma[i, :] + self.__minimum
                gammasum = np.sum(gammarow)
                for j in xrange(n_obs_type):
                    tmp = np.sum(gammarow[one_obs_data==j])
                    obs_new[i, j] = tmp / (gammasum)
                tmp1, obs_new[i, :] = self.__normalize(obs_new[i, :])
        if gshowtime:        
            time_end = time.time()
            print 'obs_calc:%.2f ms'%((time_end - time_start) * 100)
        #evaluation
#        prob = np.sum(alpha_mat[:, -1])
#        print obs_new[0, 2]
        
        return (prior_new, trans_new, obs_new)
        
        
    def evaluate(self, obs_data):
#        print 'evaluation...'
        n_data = len(obs_data)
        prob_all = np.zeros(n_data, dtype=np.float32)
        for i in xrange(n_data):
            alpha_mat, scales = self.__foreward_mat_one(obs_data[i])
            prob_all[i] = np.sum(np.log(scales))
#        prob_mean = np.sum(prob_all) / n_data
#        prob_log = np.log(prob_mean)
        prob_log = np.sum(prob_all)
        
        return prob_log
    
    
    def evaluate_each(self, obs_data):
#        print 'evaluation...'
        n_data = len(obs_data)
        prob_all = np.zeros(n_data, dtype=np.float32)
        for i in xrange(n_data):
            alpha_mat, scales = self.__foreward_mat_one(obs_data[i])
#            print alpha_mat
            prob_all[i] = np.sum(np.log(scales))
        prob_log = prob_all
        
        return prob_log
    
    
    def save(self, fn):
        f = open(fn, 'wb')
        cPickle.dump((self.__prior_state, self.__trans_mat, self.__obs_mat), f)
        f.close()
        print 'save...'
    
    
    def read(self, fn):
        f = open(fn, 'rb')
#        if f is None:
#            print 'open', fn, 'is failed.'
#            return (None, None, None)
        params = cPickle.load(f)
        f.close()
        self.__prior_state = params[0]
        self.__trans_mat = params[1]
        self.__obs_mat = params[2]
        
        self.__n_state = self.__prior_state.shape[0]
        self.__n_obsve = self.__obs_mat.shape[1]
        
        return (self.__prior_state, self.__trans_mat, self.__obs_mat)
        
        
    def getparams(self):
        """
        return (prior_state, trans_mat, obs_mat)
        """
        return (self.__prior_state, self.__trans_mat, self.__obs_mat)
        
        
    def __train_iter(self, obs_data, state_chain = None):
        
        n_obs = len(obs_data)
        prior_new = np.zeros_like(self.__prior_state)
        trans_new = np.zeros_like(self.__trans_mat)
        obs_new = np.zeros_like(self.__obs_mat)
#        probs = 0
        
        if state_chain != None:
            for i in xrange(n_obs):
                
                if gshowtime:
                    time_start = time.time()
                
                prior, trans, obs = self.__train_one(obs_data[i], state_chain[i])
                
#                probs += prob
                prior_new += prior
                trans_new += trans
                obs_new += obs
                
                if gshowtime:
                    time_end = time.time()
                    print '__train_one:%.2f ms'%((time_end - time_start) * 100)
        else:
            for i in xrange(n_obs):
                prior, trans, obs = self.__train_one(obs_data[i])
#                print i, ':', prior
#                print i, ':', prob
#                print i, ':', trans
                
                prior_new += prior
                trans_new += trans
                obs_new += obs
                
        
        prior_new /= n_obs
        trans_new /= n_obs
        obs_new /= n_obs
#        probs /= n_obs
        
        
        
        return (prior_new, trans_new, obs_new)
        
        
    def train(self, obs_data, state_chain = None, iter_max = 10):
        """
        obs_data: it is a list object, it's elements are numpy array objects and maybe have different size
        state_chain: it is a list object, it's elements are numpy array objects and maybe have different size
        """
        
        if state_chain is None:
            print
            print '----====No state_chain mode.====----'
            print
        else:
            print
            print '----====With state_chain mode.====----'
            print
#        print 'train...'
#        prior_old = self.__prior_state
#        trans_old = self.__trans_mat
#        obs_old = self.__obs_mat
        prob_mean = self.evaluate(obs_data)
        print 'total probabillity(%d/%d):%f'%(0, iter_max, prob_mean)
        
#        print
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
            print 'total probabillity(%d/%d):%f'%(i + 1, iter_max, prob_mean)
            
#            print
#            print prob_mean
            prob_old = prob_mean
#            break
#        print obs_new
#        print 'trans_new:'
#        print trans_new
        
    
    def viterbi_one(self, data_one):
#        print 'viterbi_one...'
#        print data_one.shape[0]
        data_len = data_one.shape[0]
#        l_prior = np.log(np.maximum(self.__prior_state, self.__minimum))
#        l_trans = np.log(np.maximum(self.__trans_mat, self.__minimum))
#        l_obs = np.log(np.maximum(self.__obs_mat, self.__minimum))
        
        l_prior = np.log(self.__prior_state + self.__minimum)
        l_trans = np.log(self.__trans_mat + self.__minimum)
        l_obs = np.log(self.__obs_mat + self.__minimum)
        
        prob_net = np.zeros((self.__n_state, data_len), dtype=np.float32)
        path_net = np.zeros((self.__n_state, data_len), dtype=np.int32)
        prob_net[:, 0] = l_prior + l_obs[:, data_one[0]]
        for i in xrange(1, data_len):
            ppre = prob_net[:, i-1]
            pnow = prob_net[:, i]
            pathnow = path_net[:, i]
            for j in xrange(self.__n_state):
                tmp = ppre + l_trans[:, j] + l_obs[j, data_one[i]]
                pnow[j] = np.max(tmp)
                pathnow[j] = np.argmax(tmp)
        
#        print 'prob:', prob_net[:, 0]
#        print 'path:', path_net
        
        state_chain = np.zeros(data_len, dtype=np.int32)
        score = np.max(prob_net[:, -1])
        state_chain[-1] = np.argmax(prob_net[:, -1], axis=0)
        for i in xrange(data_len-2, -1, -1):
            state_chain[i] = path_net[state_chain[i+1], i+1]
        
        return state_chain, score
        
        
    def viterbi(self, data):
#        print 'viterbi...'
        states = list()
        for s in data:
#            print s
            states.append(self.viterbi_one(s))
        
        return states
        


