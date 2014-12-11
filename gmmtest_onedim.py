# -*- coding: utf-8 -*-
"""
Created on Tue Nov 11 11:38:10 2014

@author: mzhang
"""


import numpy as np
import lpfunctions as lpfuncs
import cv2 as cv
import nhmmgsm1dim as nhg1d
import pylab as pl


maxiter = 20
neednum = 20
hmmfile = 'lphmm_' + str(maxiter) + '_' + str(neednum) + '_' + '_2.bin'
folderpath = '/Users/mzhang/work/LP Data2/'


class siGSM1D:
    def __init__(self, mean=0.0, variance=0.0):
        self.__min_var = 1
        self.__mean = mean
        self.__variance = variance
        if variance < self.__min_var:
            self.__variance = self.__min_var
        self.__invtmp = 1.0 / (np.sqrt(2 * np.pi * self.__variance))
    
    
    def calcProbability(self, x):
        dif = x - self.__mean
        tmp1 = -dif * dif / (2 * self.__variance)
        p = np.exp(tmp1) * self.__invtmp
        
        return p
    
    
    def setParams(self, mean=0.0, variance=0.0):
        self.__mean = mean
        self.__variance = variance
        if variance < self.__min_var:
            self.__variance = self.__min_var
        self.__invtmp = 1.0 / (np.sqrt(2 * np.pi * self.__variance))
    
    
    def getParams(self):
        return self.__mean, self.__variance
    


class siGMM1D:
    def __init__(self, gmlist=[], wgtlist=[]):
        self.__gmlist = gmlist
        self.__wgtlist = wgtlist
        self.__gmnum = len(gmlist)
        if len(gmlist) != len(wgtlist):
            print 'siGMM1D.__init__: error!!!!'
            exit()
    
    def calcProbability(self, x):
        p = 0.0
        for gm, wgt in zip(self.__gmlist, self.__wgtlist):
            p += wgt * gm.calcProbability(x)
        return p
    
    
    def calcProbabilityOfSeries(self, obsv):
        pall = 0.0
        for x in obsv:
            pall += self.calcProbability(x)
        obslen = len(obsv)
        return pall / obslen
    
    
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
        obslen = len(obsv)
        
        #calc posterior
        postlist = []
        for i in xrange(obslen):
            x = obsv[i]
            post = self.calcPosterior(x)
            postlist.append(post)
        
        #calc new mean and variance
        gmwgt_list = []
        for j in xrange(self.__gmnum):
            #calc mean
            mean_new = 0
            wgt_all = 0
            for i in xrange(obslen):
                x = obsv[i]
                post = postlist[i]
                mean_new += x * post[j]
                wgt_all += post[j]
            mean_new /= wgt_all
            
            
            gmwgt_list.append(wgt_all)
            
            #clac variance
            var_new = 0
            for i in xrange(obslen):
                x = obsv[i]
                dif = x - mean_new
                post = postlist[i]
                var_new += dif * dif * post[j]
            var_new /= wgt_all
            
            self.__gmlist[j].setParams(mean_new, var_new)
        
        #update weighgt for each gaussian model
        wgtsum = obslen #sum(gmwgt_list)
#        print 'obslen:', obslen, 'wgtsum:', wgtsum
        for j in xrange(self.__gmnum):
            self.__wgtlist[j] = gmwgt_list[j] / wgtsum
        
        
        
    def train(self, obsv, maxiter=10):
        pall = self.calcProbabilityOfSeries(obsv)
        print '%d->[%.6f], '%(-1, pall)
#        exit()
        for i in xrange(maxiter):
            self.__train_iter(obsv)
            pall = self.calcProbabilityOfSeries(obsv)
            print '%d->[%.6f], '%(i, pall),
            for j in xrange(self.__gmnum):
                gm = self.__gmlist[j]
                mean, var = gm.getParams()
                wgt = self.__wgtlist[j]
                print '%d:%.2f, %.2f, %.2f;  '%(j, mean, var, wgt), 
            print
            

def get_mean_var(datavec):
    mean = np.mean(datavec)
    var = np.var(datavec)
    
    return mean, var


def print_info_0(datavec):
    mean0, var0 = get_mean_var(datavec)
    histbins = xrange(0, 256)
    hist0, histmark0 = np.histogram(datavec, bins=histbins)
    print '--------------------'
    print mean0, var0
    print hist0
    pl.hist(datavec, bins=histbins)
    pl.show()
    

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


def initGMM1DParams(data, gmnum=2):
    datalen = len(data)
    partlen = datalen / gmnum
    gsmlist = []
    wgtlist = []
    initwgt = 1.0 / gmnum
    for i in xrange(gmnum):
        partdata = data[i * partlen:(i + 1) * partlen]
        mean, var = get_mean_var(partdata)
        gsm = siGSM1D(mean, var)
        gsmlist.append(gsm)
        wgtlist.append(initwgt)
        
    return gsmlist, wgtlist


def initGMM1DParams_2(data, gmnum=2):
    gsmlist = []
    wgtlist = []
    
    initwgt = 1.0 / gmnum
    gsm = siGSM1D(5, 400)
    gsmlist.append(gsm)
    wgtlist.append(initwgt)
    
    gsm = siGSM1D(19, 400)
    gsmlist.append(gsm)
    wgtlist.append(initwgt)
        
    return gsmlist, wgtlist


def initGMM1DParams_3(meanlist, varlist):
    gsmlist = []
    wgtlist = []
    gmnum = len(meanlist)
    initwgt = 1.0 / gmnum
    
    for i in xrange(gmnum):
        gsm = siGSM1D(meanlist[i], varlist[i])
        gsmlist.append(gsm)
        wgtlist.append(initwgt)
    
        
    return gsmlist, wgtlist
    

def generateGaussianData_0_255(meanlist, varlist, maxiter=1000):
    gnum = len(meanlist)
    datalist = []
    for i in xrange(maxiter):
        x = np.random.randint(0, 256, 1)
        for mi in xrange(gnum):
            mean = meanlist[mi]
            var = varlist[mi]
            diff = x - mean
            prob = np.exp(-diff * diff / var)
            accp = np.random.rand()
            if  accp < prob:
                datalist.append(x)
    
    datas = np.asarray(datalist)
    return datas
    

maxiter = 50

lpinfo_list = lpfuncs.getall_lps(folderpath, neednum)
datadict = getTrainingDatas(lpinfo_list)
data0 = datadict[0]

#data0 = generateGaussianData_0_255([100, 200], [200, 800], 100000)

#data1 = datadict[1]

#print_info_0(data1)
#exit()
gsmlist, wgtlist = initGMM1DParams(data0, 2)
#gsmlist, wgtlist = initGMM1DParams_3([100, 200], [200, 800])
gmm0 = siGMM1D(gsmlist, wgtlist)
gmm0.train(data0, maxiter)
print_info_0(data0)

