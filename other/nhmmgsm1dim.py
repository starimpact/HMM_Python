# -*- coding: utf-8 -*-
"""
Created on Mon Nov 10 10:41:40 2014

@author: mzhang
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov  5 14:40:30 2014

@author: mzhang

normalized each layer of hmm, the obvervation is subjected to gaussian single model(gsm)

one dimention:
fx = exp[-1/2*(x-miu)^2/(sigma)]/[sqrt(2*pi*sigma)]

n dimentions:
fx = exp[-1/2*(x-miu)'*inv(sigma)*(x-miu)]/[(2*pi)^(n/2)*sqrt(det(sigma))]

"""

import numpy as np
import cPickle
import time
import functions as funcs

gshowtime = False


#gaussian single model
class siGSM1D:
    def __init__(self):
        self.__mean = None # nx1 mean vector
        self.__sigma = None # nxn convariance matrix
        self.__factor = 1.0 # 1/[(2*pi)^(n/2)*sqrt(det(sigma))]
        self.__invsigma = None #inverse of sigma
    
    
    def __calcFactor_InvSigma(self):
        self.__invsigma = 1 / self.__sigma
        
    
    def setParams(self, mean, sigma):
        self.__mean = mean # nx1 mean vector
        if sigma < 1.0:
            sigma = 1.0
        self.__sigma = sigma # nxn convariance matrix
        self.__factor = 1.0 # 1/[(2*pi)^(n/2)*sqrt(det(sigma))]
        self.__calcFactor_InvSigma()
        
    
    def getParams(self):
        return self.__mean, self.__sigma
        
    
    def calcProbability(self, obsvec):
        dif = self.__mean - obsvec
        tmp1 = dif * self.__invsigma
        tmp2 = -tmp1 * dif / 2
        tmp3 = np.exp(tmp2)
        
        p = self.__factor * tmp3
#        p = tmp
        
        return p


#nomalized layer of hidden markov model with gaussian single model
class siNHMMGSM1D:
    def __init__(self, prior_state=None, trans_mat=None, gsm_list=None):
        self.__minimum = 1e-32
        
        if prior_state is not None:
            self.__prior_state = prior_state #a vector, n_state
            self.__n_state = prior_state.shape[0]
        if trans_mat is not None:
            self.__trans_mat = trans_mat  #transition probability matrix, n_state X n_state
        if gsm_list is not None:
            self.__gsm_list = gsm_list  #observation probability follows gaussian model
        self.__allProbability = None
    
    
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
        
        
    def __foreward_mat_one(self, one_obs_data, allprobs):
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
            ap = allprobs[i]
            obs_prob = ap[0]
                
            alpha_mat[i, 0] = self.__prior_state[i] * obs_prob
#            print self.__obs_mat[i, one_obs_data[0]], self.__prior_state[i]
        scales[0], alpha_mat[:, 0] = self.__normalize(alpha_mat[:, 0])
        
        
        for ti in xrange(1, data_len):
            for j in xrange(n_state):
                ap = allprobs[j]
                obs_prob = ap[ti]
                alpha_mat[j, ti] = np.dot(alpha_mat[:, ti-1], self.__trans_mat[:, j]) * obs_prob
            scales[ti], alpha_mat[:, ti] = self.__normalize(alpha_mat[:, ti])
#            print ti, alpha_mat[:, ti]
        
#        funcs.siPrintArray2D('%.2f, ', alpha_mat)
#        exit()
        
        return alpha_mat, scales


    def __backward_mat_one(self, one_obs_data, allprobs):
        """
        create beta matrix for a serial data: number of state by length of one observation data
        
        one_obs_data: one serial observation data
        """
#        print 'backward matrix....'
        data_len = one_obs_data.shape[0]
        n_state = self.__n_state
        beta_mat = np.zeros((n_state, data_len), dtype = np.float32)
        obs_prob = np.zeros(n_state, dtype=np.float32)
        beta_mat[:, -1] = 1.0
        for ti in xrange(data_len-2, -1, -1):
            for j in xrange(n_state):
                ap = allprobs[j]
                obs_prob[j] = ap[ti+1]
                
            for j in xrange(n_state):
                beta_mat[j, ti] = np.dot(beta_mat[:, ti+1], self.__trans_mat[j, :] * obs_prob)
            scale, beta_mat[:, ti] = self.__normalize(beta_mat[:, ti])
        
        return beta_mat
    
    
    def __train_one(self, one_obs_data, allprobs):
        """
        train on one serial data
        state_path: state chain
        """
        
#        print 'obs_shape:', one_obs_data.shape
        n_state = self.__n_state
        obs_len = one_obs_data.shape[0]
        trans_mat = self.__trans_mat
        
        if gshowtime:
            time_start = time.time()
        
        alpha_mat, scales = self.__foreward_mat_one(one_obs_data, allprobs)
        beta_mat = self.__backward_mat_one(one_obs_data, allprobs)
        
        if gshowtime:
            time_end = time.time()
            print 'alpha_beta_calc:%.2f ms'%((time_end - time_start) * 100)
        
#        print 'alpha:', alpha_mat.shape
#        print 'beta:', beta_mat.shape
        #calc epsilon matrix
        if gshowtime:
            time_start = time.time()
        
        eps_list = []
        for ti in xrange(obs_len-1):
            epsilon = np.zeros((n_state, n_state), dtype=np.float32)
            for i in xrange(n_state):
                for j in xrange(n_state):
                    obs_prob = allprobs[j][ti+1]
                    epsilon[i, j] = alpha_mat[i, ti] * trans_mat[i, j] * obs_prob * beta_mat[j, ti+1]
        
            sumeps = np.sum(epsilon)
            if sumeps < self.__minimum: # handle zero problem
                epsilon += self.__minimum
                sumeps = np.sum(epsilon)
        
            epsilon /= sumeps
            eps_list.append(epsilon)
        
        if gshowtime:
            time_end = time.time()
            print 'epsilon_calc:%.2f ms'%((time_end - time_start) * 100)
        
        #calc gamma matrix
        gamma = alpha_mat * beta_mat
#        print 'gamma:', gamma.shape
        
        for ti in xrange(obs_len):
            gamma_tmp = gamma[:, ti]
            tmp2, gamma[:, ti] = self.__normalize(gamma_tmp)
#        print gamma
        
#        exit()
        #calc new transition matrix
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
        
        #calc new prior matrix
        prior_new = gamma[:, 0]
        tmp1, prior_new = self.__normalize(prior_new)
#        print prior_new
        
        
        return (prior_new, trans_new, gamma)
        
        
    def evaluate(self, obs_data):
#        print 'evaluation...'
        n_data = len(obs_data)
        prob_all = np.zeros(n_data, dtype=np.float32)
        for i in xrange(n_data):
            alpha_mat, scales = self.__foreward_mat_one(obs_data[i], self.__allProbability[i])
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
        cPickle.dump((self.__prior_state, self.__trans_mat, self.__gsm_list), f)
        f.close()
        print 'save...'
    
    
    def read(self, fn):
        f = open(fn, 'rb')
        params = cPickle.load(f)
        f.close()
        self.__prior_state = params[0]
        self.__trans_mat = params[1]
        self.__gsm_list = params[2]
        
        self.__n_state = self.__prior_state.shape[0]
        
        return (self.__prior_state, self.__trans_mat, self.__gsm_list)
        
        
    def getparams(self):
        """
        return (prior_state, trans_mat, gsm_list)
        """
        return (self.__prior_state, self.__trans_mat, self.__gsm_list)
        
        
    def __updategsms(self, obs_data, gamma_list):
        n_obs = len(obs_data)
        
#        print gamma_list[1][:, 0]
#        print np.sum(gamma_list[0], axis=0)
        
        gsm_list_new = []
        #estimate new module
        for si in xrange(self.__n_state):
            #calc new mean
            meanall = 0.0
            wgtall = 0
#            print si
            for i in xrange(n_obs):
                obs_one = obs_data[i]
                gamma_one = gamma_list[i]
#                print 'gamma_one:', np.sum(gamma_one)
#                print gamma_one.shape
                gamma_row = gamma_one[si, :]
                meanall += np.sum(gamma_row * obs_one)
                wgtall += np.sum(gamma_row)
            mean_new = meanall / wgtall
            
            #calc new variance
            sigmaall = 0.0
            wgtall = 0
            for i in xrange(n_obs):
                obs_one = obs_data[i]
                gamma_one = gamma_list[i]
                gamma_row = gamma_one[si, :]
                diffs = obs_one - mean_new
                diffs_2 = diffs * diffs
                sigmaall += np.sum(gamma_row * diffs_2)
                wgtall += np.sum(gamma_row)
            sigma_new = sigmaall / wgtall
#            print 'wgtall2:', wgtall
            if si == 0:
                print '%3d:%6.1f %6.1f'%(si, mean_new, sigma_new)
            
            gsm_new = siGSM1D()
            gsm_new.setParams(mean_new, sigma_new)
            gsm_list_new.append(gsm_new)
        
        return gsm_list_new
        
        
    def __train_iter(self, obs_data):
        n_obs = len(obs_data)
        prior_new = np.zeros_like(self.__prior_state)
        trans_new = np.zeros_like(self.__trans_mat)
        gamma_list = []
        if gshowtime:
            time_start = time.time()
        
        for i in xrange(n_obs):
            prior, trans, gamma = self.__train_one(obs_data[i], self.__allProbability[i])
            gamma_list.append(gamma)
            prior_new += prior
            trans_new += trans
        
        prior_new /= n_obs
        trans_new /= n_obs
        gsm_list_new = self.__updategsms(obs_data, gamma_list)
        
        if gshowtime:
            time_end = time.time()
            print '__train_one:%.2f ms'%((time_end - time_start) * 100)
        
        
        return (prior_new, trans_new, gsm_list_new)
        
        
    def train(self, obs_data, iter_max = 10):
        """
        obs_data: it is a list object, it's elements are numpy array objects and maybe have different size
        """
        
        for i in xrange(iter_max):
            
            self.__allProbability = []
            for oi in xrange(len(obs_data)):
                allprob = []
                for api in xrange(self.__n_state):
                    gsm_one = self.__gsm_list[api]
                    allprobone = []
                    one_obs_data = obs_data[oi]
                    obs_len = one_obs_data.shape[0]
                    for ti in xrange(obs_len):
                        one_vec = one_obs_data[ti] 
                        obs_prob = gsm_one.calcProbability(one_vec)
                        allprobone.append(obs_prob)
                    allprob.append(allprobone)
                self.__allProbability.append(allprob)
            
            prob_mean = self.evaluate(obs_data)
            print 'total probabillity(%d/%d):%f'%(i + 1, iter_max, prob_mean)
#            exit()
            prior_new, trans_new, gsm_list_new = self.__train_iter(obs_data)
            
            np.copyto(self.__prior_state, prior_new)
            np.copyto(self.__trans_mat, trans_new)
            self.__gsm_list = gsm_list_new
            
            
        
    def __calcObsOfEachState(self, obs):
        obsvals = np.zeros(self.__n_state, dtype=np.float32)
        for i in xrange(self.__n_state):
            gsm = self.__gsm_list[i]
            obsvals[i] = gsm.calcProbability(obs)
        return obsvals
        
        
    def viterbi_one(self, data_one):
#        print 'viterbi_one...'
#        print data_one.shape[0]
        data_len = data_one.shape[0]
        
        l_prior = np.log(self.__prior_state + self.__minimum)
        l_trans = np.log(self.__trans_mat + self.__minimum)
        
        prob_net = np.zeros((self.__n_state, data_len), dtype=np.float32)
        path_net = np.zeros((self.__n_state, data_len), dtype=np.int32)
        prob_net[:, 0] = l_prior + np.log(self.__calcObsOfEachState(data_one[0]) + self.__minimum)
        for i in xrange(1, data_len):
            ppre = prob_net[:, i-1]
            pnow = prob_net[:, i]
            pathnow = path_net[:, i]
#            print 'obsofsate:', self.__calcObsOfEachState(data_one[i])
            l_obsone = np.log(self.__calcObsOfEachState(data_one[i]) + self.__minimum)
            for j in xrange(self.__n_state):
                tmp = ppre + l_trans[:, j] + l_obsone[j]
                pnow[j] = np.max(tmp)
                pathnow[j] = np.argmax(tmp)
                
#            exit()
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
        


