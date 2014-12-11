# -*- coding: utf-8 -*-
"""
Created on Tue Nov 18 15:55:32 2014

@author: mzhang
"""


import numpy as np
import cPickle
import time
import functions as funcs

gshowtime = False


#gaussian single model
class siLogGSM:
    def __fullRankCoVariance(self):
        u, s, v = np.linalg.svd(self.__covariance)
        for i in xrange(len(s)):
            s[i] = s[i] if s[i] > self.__minvar else self.__minvar
        us = np.dot(u, np.diag(s))
        usv = np.dot(us, u.T)
        self.__covariance = usv
    
    def __calcFactor_InvConVariance(self):
        dim_num = len(self.__mean)
        detcov = np.linalg.det(self.__covariance)
        tmp1 = -0.5 * (np.log(detcov) + np.log(2) + dim_num * np.log(np.pi))
        self.__factor = tmp1
        self.__invconvariance = np.linalg.inv(self.__covariance)
        
    def __init__(self, mean, covariance):
        self.__minvar = 1e-5
        self.__mean = mean
        self.__covariance = covariance
        self.__fullRankCoVariance()
        self.__calcFactor_InvConVariance()
    
    def calcLogProbability(self, x):
        dif = self.__mean - x
#        print 'calcProbability:', self.__mean.shape, x.shape
        dif = dif.reshape(len(dif), 1)
        tmp1 = np.dot(dif.T, self.__invconvariance)
        tmp2 = -np.dot(tmp1, dif) * 0.5
        p = self.__factor + tmp2
        p = p[0, 0]
#        p = 0.0
        
        return p
        
    def calcProbability(self, x):
        dif = self.__mean - x
#        print 'calcProbability:', self.__mean.shape, x.shape
        dif = dif.reshape(len(dif), 1)
        tmp1 = np.dot(dif.T, self.__invconvariance)
        tmp2 = -np.dot(tmp1, dif) / 2
        tmp3 = np.exp(tmp2)
        p = np.exp(self.__factor) * tmp3
        p = p[0, 0]
        
        return p
    
    def setParams(self, mean=0.0, variance=0.0):
        self.__mean = mean
        self.__variance = variance
        self.__fullRankCoVariance()
        self.__calcFactor_InvConVariance()
    
    def getParams(self):
        return self.__mean, self.__covariance
    
    
    def getTXT(self, prestr):
        paramList = []
        #save mean
        meanlen = len(self.__mean)
        paramList.append('gafMean_%s'%(prestr))
        paramTXT = 'extern const float gafMean_%s[%d];\n'%(prestr, meanlen)
        gsmTXT = 'const float gafMean_%s[%d] = '%(prestr, meanlen)
        gsmTXT += '{'
        for mi in xrange(meanlen):
            gsmTXT += '%.5ff'%(self.__mean[mi])
            if mi < meanlen-1:
                gsmTXT += ', '
        gsmTXT += '};\n\n'
        
        #save factor
        paramList.append('gfFactor_%s'%(prestr))
        paramTXT += 'extern const float gfFactor_%s;\n'%(prestr)
        gsmTXT += 'const float gfFactor_%s = %.5ff;\n\n'%(prestr, self.__factor)
        
        #save invcovariance
        paramList.append('gafInvCovar_%s'%(prestr))
        paramTXT += 'extern const float gafInvCovar_%s[%d];\n'%(prestr, meanlen * meanlen)
        gsmTXT += 'const float gafInvCovar_%s[%d] = '%(prestr, meanlen * meanlen)
        gsmTXT += '{'
        for cri in xrange(meanlen):
#            gsmTXT += '{'
            for cci in xrange(meanlen):
                gsmTXT += '%.5ff'%(self.__invconvariance[cri, cci])
                if cci < meanlen-1:
                    gsmTXT += ', '
#            gsmTXT += '}'
            if cri < meanlen-1:
                gsmTXT += ', '
        gsmTXT += '};\n\n'
        
        return gsmTXT, paramTXT, paramList


#gaussian single model
class siLogGMM:
    def __init__(self, gmlist=[], wgtlist=[]):
        self.__gmlist = gmlist
        self.__wgtlist = wgtlist
        self.__logwgtlist = []
        for wgt in wgtlist:
            self.__logwgtlist.append(np.log(wgt))
        self.__gmnum = len(gmlist)
        self.__minimum = 1e-32
        self.__maximum = 1e32
        if len(gmlist) != len(wgtlist):
            print 'siGMM1D.__init__: error!!!!'
            exit()
    
    def calcPosteriorByLog(self, x):
        postall = []
        log_problist = []
        log_wgtlist = self.__logwgtlist
        for gm in self.__gmlist:
            logprob = gm.calcLogProbability(x)
            log_problist.append(logprob)
            
        log_probarray = np.asarray(log_problist)
        log_wgtarray = np.asanyarray(log_wgtlist) #????????anyarray?????
        log_probwgt = log_probarray + log_wgtarray
        for logp1, logw1 in zip(log_problist, log_wgtlist):
            tmp3 = self.__minimum
            rlts = log_probwgt - (logp1 + logw1)
            rltmax = np.max(rlts)
            if rltmax < 64:
                tmp1 = np.sum(np.exp(rlts))
                tmp3 = 1 / tmp1
            postall.append(tmp3)
        postall = np.asarray(postall)
        
        return postall, log_problist, log_wgtlist
    
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
    
    def getTXT(self, prestr):
        gmnum = self.__gmnum
        paramList = []
        #save wgtlist
        paramList.append('gafWgtList_%s'%(prestr))
        paramTXT = 'extern const float gafWgtList_%s[%d];\n'%(prestr, gmnum)
        gmtxt = 'const float gafWgtList_%s[%d] = '%(prestr, gmnum)
        gmtxt += '{'
        for gi in xrange(gmnum):
            gmtxt += '%.5ff'%(self.__wgtlist[gi])
            if gi < gmnum-1:
                gmtxt += ', '
        gmtxt += '};\n\n'
        
        #save gmlist
        gmmTXTList = []
        for gi in xrange(gmnum):
            gsm = self.__gmlist[gi]
            gsmTXT, gsmparamTXT, gsmparamList = gsm.getTXT(prestr + '_' + str(gi))
            gmtxt += gsmTXT
            paramTXT += gsmparamTXT
            gmmTXTList.append(gsmparamList)
        paramList.append(gmmTXTList)
            
        return gmtxt, paramTXT, paramList
    
    
    def calcLogProbability(self, x):
        logp = 0.0
        postlist, log_problist, log_wgtlist = self.calcPosteriorByLog(x)
        
        log_postlist = np.log(postlist)
        for post, logpost, logprob, logwgt in zip(postlist, log_postlist, log_problist, log_wgtlist):
            logp += post * (logprob + logwgt - logpost)
        return logp
    
    
#    def calcEachProbability(self, x):
#        problist = []
#        for gm, wgt in zip(self.__gmlist, self.__wgtlist):
#            p = wgt * gm.calcProbability(x)
#            problist.append(p)
#        probarray = np.asarray(problist)
#        return probarray
    
    
    def calcLogProbabilityOfSeries(self, obsv):
        obslen = obsv.shape[1]
        pall = 0.0
        for oi in xrange(obslen):
            x = obsv[:, oi]
            pall += self.calcLogProbability(x)
        obslen = len(obsv)
        return pall / obslen
    
    
    def calcProbability(self, x):
        p = 0.0
        for gm, wgt in zip(self.__gmlist, self.__wgtlist):
            pp = gm.calcProbability(x)
#            print wgt, pp
            p += wgt * pp
#        print 'hi:', p
        return p
    
    
    def calcEachProbability(self, x):
        problist = []
        for gm, wgt in zip(self.__gmlist, self.__wgtlist):
            p = wgt * gm.calcProbability(x)
            problist.append(p)
        probarray = np.asarray(problist)
        return probarray
    
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
#        print obsv.flags
        #calc posterior
#        tms = time.time()
        proballold = 0.0
        postlist = []
        for i in xrange(obslen):
            x = obsv[:, i]
            post, log_prob, log_wgt = self.calcPosteriorByLog(x)
            postlist.append(post)
            
            log_post = np.log(post)
            alog_post = np.asarray(log_post)
            apost = np.asarray(post)
            alog_prob = np.asarray(log_prob)
            alog_wgt = np.asarray(log_wgt)
            proballold += np.sum(apost * (alog_prob + alog_wgt - alog_post))
            
        proballold / obslen
            
#        tme = time.time()
#        print 'calcpost:', tme - tms
#        exit()
        #calc new mean and variance
#        tms = time.time()
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
#        tme = time.time()
#        print 'calcmeancov:', tme - tms
        
        #update weighgt for each gaussian model
        wgtsum = obslen #sum(gmwgt_list)
#        print 'obslen:', obslen, 'wgtsum:', wgtsum
        for j in xrange(self.__gmnum):
            self.__wgtlist[j] = gmwgt_list[j] / wgtsum
            self.__logwgtlist[j] = np.log(self.__wgtlist[j])
        
        return proballold
        
    
    def train(self, obsv, maxiter=10):
#        tms = time.time()
#        pallold = self.calcLogProbabilityOfSeries(obsv)
#        tme = time.time()
#        print '%d->[%.12f], '%(-1, pallold), tme-tms
        pallold = 0.0
        for i in xrange(maxiter):
            tms = time.time()
            pallnew = self.__train_iter(obsv)
            tme = time.time()
            print 'iter %d -> probold:%.4f cost:%.3f'%(i, pallnew, tme-tms)
#            pallnew = self.calcLogProbabilityOfSeries(obsv)
#            print '%d->[%.12f], '%(i, pallnew)
            if np.abs(pallnew - pallold)<1e-3:
                print 'convergenced...'
                break
            pallold = pallnew
            


#nomalized layer of hidden markov model with gaussian single model
class siLogNHMMGMM:
    def __init__(self, prior_state=None, trans_mat=None, gmm_list=None):
        self.__minimum = 1e-32
        self.__logminimum = -1024 * 16
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
    
    
    def saveTXT(self, fn):
        paramList = []
        #write prior_sate
        pslen = len(self.__prior_state)
        paramList.append('gafPriorState')
        paramTXT = 'extern const float gafPriorState[%d];\n'%(pslen)
        gmmTXT = 'const float gafPriorState[%d] = '%(pslen)
        gmmTXT += '{'
        for pi in xrange(pslen):
            gmmTXT += '%.5ff'%self.__prior_state[pi]
            if pi < pslen-1:
                gmmTXT += ', '
        gmmTXT += '};\n\n';
        
        #write trans_mat
        paramList.append('gafTransmat')
        tmh, tmw = self.__trans_mat.shape
        paramTXT += 'extern const float gafTransmat[%d];\n'%(tmh * tmw)
        gmmTXT += 'const float gafTransmat[%d] = '%(tmh * tmw)
        gmmTXT += '{'
        for ri in xrange(tmh):
#            gmmTXT += '{'
            for ci in xrange(tmw):
                gmmTXT += '%.5ff'%(self.__trans_mat[ri, ci])
                if ci < tmw-1:
                    gmmTXT += ', '
#            gmmTXT += '}'
            if ri < tmh-1:
                gmmTXT += ', '
        gmmTXT += '};\n\n'
        
        #write gmm_list
        gmmTXTList = []
        gmmnum = len(self.__gmm_list)
        for gmmi in xrange(gmmnum):
            gmm = self.__gmm_list[gmmi]
            gmmtxttmp, gmmparamtxt, gmmparamList = gmm.getTXT(str(gmmi))
            gmmTXT += gmmtxttmp
            paramTXT += gmmparamtxt
            gmmTXTList.append(gmmparamList)
        paramList.append(gmmTXTList)
        
        fp = open(fn, 'w')
        fp.write(gmmTXT)
        fp.close()
        print 'siLogNHMMGMM is save into ', fn, '.'
        
        fn2 = 'param_' + fn
        fp = open(fn2, 'w')
        fp.write(paramTXT)
        fp.close()
        print 'siLogNHMMGMM is save into ', fn2, '.'
        
        
        setparamTXT = '{'
        setparamTXT += paramList[0] + ', ' + paramList[1] + ', '
        paramList2 = paramList[2]
#        print paramList2
        len2 = len(paramList2)
        setparamTXT += '{'
        for i2 in xrange(len2):
            subList = paramList2[i2]
            setparamTXT += '{' + subList[0] + ', '
            paramList3 = subList[1]
            len3 = len(paramList3)
            setparamTXT += '{'
            for i3 in xrange(len3):
                subList2 = paramList3[i3]
                setparamTXT += '{' + subList2[0] + ', ' + subList2[1] + ', ' + subList2[2] + '}'
                if i3 < len3-1:
                    setparamTXT += ', '
            setparamTXT += '}}'
            if i2 < len2-1:
                setparamTXT += ', '
        setparamTXT += '}};\n'
        fn3 = 'setparam_' + fn
        fp = open(fn3, 'w')
        fp.write(setparamTXT)
        fp.close()
        print 'siLogNHMMGMM is save into ', fn3, '.'
    
    
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
                
                gsm_new = siLogGSM()
                gsm_new.setParams(mean_new, cov_new)
                gmlist.append(gsm_new)
                
                wgtlist.append(wgtall)
                
            wgtall = np.sum(wgtlist)
            for gmi in xrange(gsmnum):
                wgtlist[gmi] /= wgtall
            
            gmm_new = siLogGMM(gmlist, wgtlist)
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
        
        CAN NOT USE NOW!!!!!!!!!!!!
        
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
    
    
    def __calcLogObsOfEachState(self, obs):
        obsvals = np.zeros(self.__n_state, dtype=np.float32)
        for i in xrange(self.__n_state):
            gmm = self.__gmm_list[i]
            obsvals[i] = gmm.calcLogProbability(obs)
#            print '%.2f[%.2f],'%(obsvals[i], np.log(gmm.calcProbability(obs))),
#            print '%.2f,'%(obsvals[i]),
#        print
        
        return obsvals
    
    
    def viterbi_one(self, data_one):
#        print 'viterbi_one...'
#        print data_one.shape[0]
        data_len = data_one.shape[1]
        
        l_prior = np.log(self.__prior_state + self.__minimum)
        l_prior[self.__prior_state<self.__minimum] = self.__logminimum
#        print 'l_prior:'
#        funcs.siPrintArray2D('%7.2f', l_prior.reshape(1, self.__n_state))
        l_trans = np.log(self.__trans_mat + self.__minimum)
        l_trans[self.__trans_mat<self.__minimum] = self.__logminimum
#        print 'l_trans:'
#        funcs.siPrintArray2D('%7.2f', l_trans)
        
        prob_net = np.zeros((self.__n_state, data_len), dtype=np.float32)
        path_net = np.zeros((self.__n_state, data_len), dtype=np.int32)
        prob_net[:, 0] = l_prior + self.__calcLogObsOfEachState(data_one[:, 0])
        for i in xrange(1, data_len):
            ppre = prob_net[:, i-1]
            pnow = prob_net[:, i]
            pathnow = path_net[:, i]
#            print 'obsofsate:', self.__calcObsOfEachState(data_one[i])
            l_obsone = self.__calcLogObsOfEachState(data_one[:, i])
            for j in xrange(self.__n_state):
                tmp = ppre + l_trans[:, j] + l_obsone[j]
                pnow[j] = np.max(tmp)
                pathnow[j] = np.argmax(tmp)
        
#            exit()
#        print 'prob:', prob_net[:, 0]
#        print 'pathnet:'
#        funcs.siPrintArray2D('%4d', path_net.T)
#        
#        print 'prob_net:'
#        funcs.siPrintArray2D('%4d', prob_net.T)
        
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
        

            