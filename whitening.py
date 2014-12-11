# -*- coding: utf-8 -*-
"""
Created on Sat Jun 28 19:16:56 2014

@author: mzhang
"""

import numpy as np
from numpy import linalg
import os
import cv2
import cPickle
import ctypes

lpcr_func_c = ctypes.CDLL('./lpcr_func_c.so')

def load_image_vecs(folder_name,maxnum=-1,stdsize=(28,28)):
# return: all data
#    print folder_name
    files=os.listdir(folder_name)
#    print len(files)
    datalist=list()
    num = 0
    for fname in files:
        fullpath=folder_name+'/'+fname
        if os.path.isdir(fullpath):
            files2=os.listdir(fullpath)
            for imgname in files2:
                imgpath=fullpath+'/'+imgname
                img=cv2.imread(imgpath)
                img=img[:,:,0]
                imgrsz=cv2.resize(img,stdsize) #resize image to 28x28
                imgvec=np.reshape(imgrsz,stdsize[0]*stdsize[1])
                fimgvec=imgvec.astype(np.float32)
#                fimgvec=normalize_img_data_to_unitvar(fimgvec)
                lpcr_func_c.normalize_img_data_to_0_1(imgvec.ctypes, fimgvec.ctypes, stdsize[0]*stdsize[1], 20);
#                fimgvec-=np.mean(fimgvec)
#                print fimgvec
                datalist.append(fimgvec)
                num += 1
                if maxnum >= 0 and num == maxnum:
                    break
        if maxnum >= 0 and num == maxnum:
            break
#                cv2.imshow('rsz',imgvec.reshape((28,28)))
#                cv2.waitKey(0)
    print num, 'samples is loaded...'
    return datalist

def cent_cov(dataarray):
    #data is organized in column, zero mean in the row
    vecmean = np.mean(dataarray, 1)
    num = dataarray.shape[1]
#    print dataarray.shape, vecmean.shape
    vecarraycent = (dataarray.T - vecmean).T
    coval = vecarraycent.dot(vecarraycent.T) / num
    return coval

def normalize_img_data_to_unitvar(imgvec):
    fimgvec = imgvec.astype(np.float32)
    stdv = np.std(fimgvec)
    fimgvec -= np.mean(fimgvec)
    fimgvec /= stdv
    
    return fimgvec

def cent_cov2(dataarray):
    #data is organized in column, zero mean in the column
    vecmean = np.mean(dataarray, 0)
#    num = dataarray.shape[0]
#    print dataarray.shape, vecmean.shape
    vecarraycent = dataarray - vecmean
#    coval = vecarraycent.dot(vecarraycent.T) / num
#    print vecarraycent.shape
    coval = cent_cov(vecarraycent)
    return coval


def vec_2_img(vec, s):
    vec = vec - np.amin(vec)
    vec /= np.amax(vec)
    vec *= 255
    img = np.reshape(vec, s)
    img = img.astype(np.uint8)
    return img

def zca_white_all(vecarray, needvu, needvs):
    vecrot = needvu.T.dot(vecarray)
    #coval2 = cent_cov(vecarrayrot)
    vecrot = (vecrot.T / np.sqrt(needvs + 0.1)).T
    vec2 = needvu.dot(vecrot)
    return vec2

def zca_white_one(vecone, needvu, needvs):
    vecrot = needvu.T.dot(vecone)
    #coval2 = cent_cov(vecarrayrot)
    vecrot = (vecrot.T / np.sqrt(needvs + 0.1)).T
    vec2 = needvu.dot(vecrot)
    return vec2

def get_U_S_image(vecarray):
    coval = cent_cov2(vecarray) #used in image
        
    vu, vs, vv = linalg.svd(coval) #coval = vu * vs * vv, eigenvector is column of vu, and row of vv
    energyall = np.sum(vs)
    alldim = len(vs)
    needdim = alldim
    for ni in xrange(alldim):
        if np.sum(vs[:ni+1]) * 1000 > energyall * 990:
            needdim = ni + 1
            break
#    needdim = alldim
    print 'to keep 99.5% variance we need', needdim, 'dimensions.'
    needvu = vu[:,:needdim]
    needvs = vs[:needdim]
    return needvu, needvs

def zca_white(vecarray, covmethod = 1):
    if covmethod == 1:
        coval = cent_cov(vecarray)
    elif covmethod == 2:
        coval = cent_cov2(vecarray) #used in image
    else:
        return None
        
    vu, vs, vv = linalg.svd(coval) #coval = vu * vs * vv, eigenvector is column of vu, and row of vv
    energyall = np.sum(vs)
    alldim = len(vs)
    needdim = alldim
    for ni in xrange(alldim):
        if np.sum(vs[:ni+1]) * 1000 > energyall * 990:
            needdim = ni + 1
            break
#    needdim = alldim
    print 'to keep 99.5% variance we need', needdim, 'dimensions.'
    needvu = vu[:,:needdim]
    needvs = vs[:needdim]
    
    vecarrayrot = needvu.T.dot(vecarray)
    #coval2 = cent_cov(vecarrayrot)
    vecarrayrot = (vecarrayrot.T / np.sqrt(needvs + 0.1)).T
    
    vecarray2 = needvu.dot(vecarrayrot)
    
    return vecarray2


def test():  
    imgveclist = load_image_vecs('/media/mzhang/data/twcharset0_33')
    vecarray = np.asarray(imgveclist).T
    needvu, needvs = get_U_S_image(vecarray)
    vecarray2 = zca_white_all(vecarray, needvu, needvs)
    
    datalen = vecarray.shape[1]
    for ni in xrange(0, datalen, 10):
        vec = vecarray[:, ni]
        vec2 = vecarray2[:, ni]
    #    print vec[:20]
    #    print vec2[:20]
        
        img = vec_2_img(vec, (28, 28))
        img2 = vec_2_img(vec2, (28, 28))
        
        imgall = np.append(img, img2, axis = 1)
        
        cv2.imshow('img', imgall)
        cv2.waitKey(0)
    


