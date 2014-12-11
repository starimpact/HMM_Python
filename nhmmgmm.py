# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 13:45:14 2014

@author: mzhang
"""



import numpy as np
import cPickle
import time
import functions as funcs

gshowtime = False


#gaussian single model
class siGSM:
    def __fullRankCoVariance(self):
        u, s, v = np.linalg.svd(self.__covariance)
        for i in xrange(len(s)):
            s[i] = s[i] if s[i] > self.__minvar else self.__minvar
        us = np.dot(u, np.diag(s))
        usv = np.dot(us, u.T)
        self.__covariance = usv
    
    def __calcFactor_InvConVariance(self):
        dim_num = len(self.__mean)
        tmp1 = (2 * (np.pi ** dim_num) * np.linalg.det(self.__covariance))
        tmp2 = np.sqrt(tmp1)
        self.__factor = 1 / tmp2
        self.__invconvariance = np.linalg.inv(self.__covariance)
        
    def __init__(self, mean, covariance):
        self.__minvar = 1e-5
        self.__mean = mean
        self.__covariance = covariance
        self.__fullRankCoVariance()
        self.__calcFactor_InvConVariance()
    
    def calcProbability(self, x):
        dif = self.__mean - x
#        print 'calcProbability:', self.__mean.shape, x.shape
        dif = dif.reshape(len(dif), 1)
        tmp1 = np.dot(dif.T, self.__invconvariance)
        tmp2 = -np.dot(tmp1, dif) / 2
        tmp3 = np.exp(tmp2)
        p = self.__factor * tmp3
        p = p[0, 0]
        
        return p
    
    def setParams(self, mean=0.0, variance=0.0):
        self.__mean = mean
        self.__variance = variance
        self.__fullRankCoVariance()
        self.__calcFactor_InvConVariance()
    
    def getParams(self):
        return self.__mean, self.__covariance


#gaussian single model
class siGMM:
    def __init__(self, gmlist=[], wgtlist=[]):
        self.__gmlist = gmlist
        self.__wgtlist = wgtlist
        self.__gmnum = len(gmlist)
        if len(gmlist) != len(wgtlist):
            print 'siGMM1D.__init__: error!!!!'
            exit()
    
    
    def printInfo(self):
        i = 0
        for gm, wgt in zip(self.__gmlist, self.__wgtlist):
            mean, var = gm.getParams()
#            print 'i:', i, 'mean:', mean, 'var:', var, 'wgt:', wgt
            print '%3d:%6.1f %6.1f %6.5f'%(i, mean, var, wgt)
            i += 1
            
    
    def getGSMNum(self):
        return self.__gmnum
    
    def getGMMParams(self):
        return self.__gmlist, self.__wgtlist
        
    def printWeights(self):
        print 'weight:', 
        for wgt in self.__wgtlist:
            print '%.4f, '%(wgt),
        print
        
    
    def calcProbability(self, x):
        p = 0.0
        for gm, wgt in zip(self.__gmlist, self.__wgtlist):
            p += wgt * gm.calcProbability(x)
        return p
    
    
    def calcEachProbability(self, x):
        problist = []
        for gm, wgt in zip(self.__gmlist, self.__wgtlist):
            p = wgt * gm.calcProbability(x)
            problist.append(p)
        probarray = np.asarray(problist)
        return probarray
    
    
#    def calcProbabilityOfSeries(self, obsv):
#        pall = 0.0
#        for x in obsv:
#            pall += self.calcProbability(x)
#        obslen = len(obsv)
#        return pall / obslen
    
    
    def calcPosterior(self, x):
        postall = []
        pall = self.calcProbability(x)
        for gm, wgt in zip(self.__gmlist, self.__wgtlist):
            p = wgt * gm.calcProbability(x)
            post = p / pall
            postall.append(post)
        postall = np.asarray(postall)
        
        return postall
    
    
    def __train_iter(self, obsv):
        obslen = obsv.shape[1]
        
        #calc posterior
        postlist = []
        for i in xrange(obslen):
            x = obsv[:, i]
            post = self.calcPosterior(x)
            postlist.append(post)
        
        #calc new mean and variance
        gmwgt_list = []
        for j in xrange(self.__gmnum):
            mean_old, cov_old = self.__gmlist[j].getParams()
            
            #calc mean
            mean_new = np.zeros_like(mean_old)
            wgt_all = 0
            for i in xrange(obslen):
                x = obsv[:, i]                
                post = postlist[i]
                mean_new += x * post[j]
                wgt_all += post[j]
            mean_new /= wgt_all
            gmwgt_list.append(wgt_all)
            
            #clac covariance
            var_new = np.zeros_like(cov_old)
            for i in xrange(obslen):
                x = obsv[:, i]
                dif = x - mean_new
                dif = dif.reshape(len(dif), 1)
                post = postlist[i]
                var_new += np.dot(dif, dif.T) * post[j]
            var_new /= wgt_all
            
            self.__gmlist[j].setParams(mean_new, var_new)
        
        #update weighgt for each gaussian model
        wgtsum = obslen #sum(gmwgt_list)
#        print 'obslen:', obslen, 'wgtsum:', wgtsum
        for j in xrange(self.__gmnum):
            self.__wgtlist[j] = gmwgt_list[j] / wgtsum
        
    
    def train(self, obsv, maxiter=10):
#        pall = self.calcProbabilityOfSeries(obsv)
#        print '%d->[%.6f], '%(-1, pall)
#        exit()
        for i in xrange(maxiter):
            self.__train_iter(obsv)
#            pall = self.calcProbabilityOfSeries(obsv)
#            print '%d->[%.6f], '%(i, pall),
#            for j in xrange(self.__gmnum):
#                gm = self.__gmlist[j]
#                mean, var = gm.getParams()
#                wgt = self.__wgtlist[j]
#                print '%d:%.2f, %.2f, %.2f;  '%(j, mean, var, wgt), 
#            print
            
            
            

#nomalized layer of hidden markov model with gaussian single model
class siNHMMGMM:
    def __init__(self, prior_state=None, trans_mat=None, gmm_list=None):
        self.__minimum = 1e-32
        
        if prior_state is not None:
            self.__prior_state = prior_state #a vector, n_state
            self.__n_state = prior_state.shape[0]
        if trans_mat is not None:
            self.__trans_mat = trans_mat  #transition probability matrix, n_state X n_state
        if gmm_list is not None:
            self.__gmm_list = gmm_list  #observation probability follows gaussian model
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
            obs_prob = np.sum(allprobs[i][0])
            alpha_mat[i, 0] = self.__prior_state[i] * obs_prob
#            print self.__obs_mat[i, one_obs_data[0]], self.__prior_state[i]
        scales[0], alpha_mat[:, 0] = self.__normalize(alpha_mat[:, 0])
        
        for ti in xrange(1, data_len):
            for j in xrange(n_state):
                obs_prob = np.sum(allprobs[j][ti])
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
                obs_prob[j] = np.sum(allprobs[j][ti+1])
            
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
                    obs_prob = np.sum(allprobs[j][ti+1])
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
        cPickle.dump((self.__prior_state, self.__trans_mat, self.__gmm_list), f)
        f.close()
        print 'save...'
    
    
    def read(self, fn):
        f = open(fn, 'rb')
        params = cPickle.load(f)
        f.close()
        self.__prior_state = params[0]
        self.__trans_mat = params[1]
        self.__gmm_list = params[2]
        
        self.__n_state = self.__prior_state.shape[0]
        
        
    def getparams(self):
        """
        return (prior_state, trans_mat, gsm_list)
        """
        return (self.__prior_state, self.__trans_mat, self.__gmm_list)
    
    
    def __calcMeanCoVariance(self, obs_data, gamma_list, allprob, gmi, si):
        n_obs = len(obs_data)
        meanold, covold = self.__gmm_list[gmi].getParams()
        #calc new mean
        meanall = np.zeros_like(meanold)
        wgtall = 0
        for i in xrange(n_obs):
            obs_one = obs_data[i]
            len_obs_one = obs_one.shape[1]
            gamma_one = gamma_list[i]
#            print 'gamma_one:', np.sum(gamma_one)
            gamma_row = gamma_one[si, :]
            probobs = allprob[i][si]
#            print 'i:', i, 'len_probobs:', len(probobs), 'len_obs_one:', len_obs_one
            for oi in xrange(len_obs_one):
                probgmm = probobs[oi]
                postgsm = probgmm[gmi] / np.sum(probgmm)
                tmp1 = gamma_row[oi] * postgsm
#                tmp1 = postgsm
                meanall += tmp1 * obs_one[:, oi]
                wgtall += tmp1
        mean_new = meanall / wgtall
        
        #calc new variance
        covall = np.zeros_like(covold)
        for i in xrange(n_obs):
            obs_one = obs_data[i]
            len_obs_one = obs_one.shape[1]
            gamma_one = gamma_list[i]
            gamma_row = gamma_one[si, :]
            probobs = allprob[i][si]
            for oi in xrange(len_obs_one):
                diff = mean_new - obs_one[:, oi]
                diff = diff.reshape(len(diff), 1)
                probgmm = probobs[oi]
                postgsm = probgmm[gmi] / np.sum(probgmm)
                tmp1 = gamma_row[oi] * postgsm
#                tmp1 = postgsm
                covall += tmp1 * np.dot(diff, diff.T)
        cov_new = covall / wgtall
        
        return mean_new, cov_new, wgtall
    
    def __updategmms(self, obs_data, gamma_list):
        
        allprob = self.__allProbability
        
        gmm_list_new = []
        #estimate new module
        for si in xrange(self.__n_state):
            gmlist = []
            wgtlist = []
            gsmnum = self.__gmm_list[si].getGSMNum()
#            print 'si:', si
            for gmi in xrange(gsmnum):
#                print 'gmi:', gmi
                mean_new, cov_new, wgtall = self.__calcMeanCoVariance(obs_data, gamma_list, allprob, gmi, si)
                
                gsm_new = siGSM()
                gsm_new.setParams(mean_new, cov_new)
                gmlist.append(gsm_new)
                
                wgtlist.append(wgtall)
                
            wgtall = np.sum(wgtlist)
            for gmi in xrange(gsmnum):
                wgtlist[gmi] /= wgtall
            
            gmm_new = siGMM(gmlist, wgtlist)
            gmm_list_new.append(gmm_new)
            if si == 0 or si == 5:
                print 'gmm_%d:'%(si)
                gmm_new.printInfo()
        
        return gmm_list_new


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
        gmm_list_new = self.__updategmms(obs_data, gamma_list)
        
        
        
        if gshowtime:
            time_end = time.time()
            print '__train_one:%.2f ms'%((time_end - time_start) * 100)
        
        
        return (prior_new, trans_new, gmm_list_new)
        
        
    def train(self, obs_data, iter_max = 10):
        """
        obs_data: it is a list object, it's elements are numpy array objects and maybe have different size
        """
        
        print 'start training now...'
        for i in xrange(iter_max):
            
            self.__allProbability = []
            for oi in xrange(len(obs_data)):
                allprob = []
                for api in xrange(self.__n_state):
                    gmm_one = self.__gmm_list[api]
                    allprobone = []
                    one_obs_data = obs_data[oi]
                    obs_len = one_obs_data.shape[1]
                    for ti in xrange(obs_len):
                        one_v = one_obs_data[:, ti] #scalar
                        obs_prob = gmm_one.calcEachProbability(one_v)
                        allprobone.append(obs_prob)
                    allprob.append(allprobone)
                self.__allProbability.append(allprob)
            
            prob_mean = self.evaluate(obs_data)
            print 'total probabillity(%d/%d):%f'%(i + 1, iter_max, prob_mean)
#            exit()
            prior_new, trans_new, gmm_list_new = self.__train_iter(obs_data)
            
            np.copyto(self.__prior_state, prior_new)
            np.copyto(self.__trans_mat, trans_new)
            self.__gmm_list = gmm_list_new
    
    
    def __calcObsOfEachState(self, obs):
        obsvals = np.zeros(self.__n_state, dtype=np.float32)
        for i in xrange(self.__n_state):
            gmm = self.__gmm_list[i]
            obsvals[i] = gmm.calcProbability(obs)
        
        return obsvals
    
    
    def viterbi_one(self, data_one):
#        print 'viterbi_one...'
#        print data_one.shape[0]
        data_len = data_one.shape[1]
        
        l_prior = np.log(self.__prior_state + self.__minimum)
        l_trans = np.log(self.__trans_mat + self.__minimum)
        
        prob_net = np.zeros((self.__n_state, data_len), dtype=np.float32)
        path_net = np.zeros((self.__n_state, data_len), dtype=np.int32)
        prob_net[:, 0] = l_prior + np.log(self.__calcObsOfEachState(data_one[:, 0]) + self.__minimum)
        for i in xrange(1, data_len):
            ppre = prob_net[:, i-1]
            pnow = prob_net[:, i]
            pathnow = path_net[:, i]
#            print 'obsofsate:', self.__calcObsOfEachState(data_one[i])
            l_obsone = np.log(self.__calcObsOfEachState(data_one[:, i]) + self.__minimum)
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
        
