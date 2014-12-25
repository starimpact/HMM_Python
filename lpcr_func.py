# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 11:59:47 2014

@author: mzhang
"""

import numpy as np
import os
import cv2
import theano.tensor as T
from theano import shared
import cPickle
import ctypes
import showdata
import theano
import whitening
from ctypes import *

lpcr_func_c = ctypes.CDLL('lpcr_func_c.so')


def getsamples_scales_c(lpimg, newbbs, newchbbs, stdsize, sample_maxnum):
    lpcr_func_c.getsamples_scales.argtypes = \
        [POINTER(c_ubyte), c_int , c_int, POINTER(c_int), c_int, \
        POINTER(c_int), c_int, c_int, c_int, POINTER(c_float), POINTER(c_int), POINTER(c_int)]
    imgh, imgw = lpimg.shape
    newbbs_tmp = np.asarray(newbbs, dtype=np.int32)
    newbbs_len = newbbs_tmp.shape[0]
    newchbbs_tmp = np.asarray(newchbbs, dtype=np.int32)
    newchbbs_len = newchbbs_tmp.shape[0] * newchbbs_tmp.shape[1]
    stdh, stdw = stdsize
    sample_maxnum_tmp = np.asarray(sample_maxnum, dtype=np.int32)
    samples = np.zeros((sample_maxnum, stdh*stdw), dtype=np.float32)
    tags = np.zeros(sample_maxnum, dtype=np.int32)
    lpcr_func_c.getsamples_scales(lpimg.ctypes.data_as(POINTER(c_ubyte)), imgw, imgh, \
                        newbbs_tmp.ctypes.data_as(POINTER(c_int)), newbbs_len, \
                        newchbbs_tmp.ctypes.data_as(POINTER(c_int)), newchbbs_len, \
                        stdw, stdh, \
                        samples.ctypes.data_as(POINTER(c_float)), tags.ctypes.data_as(POINTER(c_int)), \
                        sample_maxnum_tmp.ctypes.data_as(POINTER(c_int)))
    samples_list = samples[:sample_maxnum_tmp, :].tolist()
    tags_list = tags[:sample_maxnum_tmp].tolist()
    
    return samples_list, tags_list


def get_3sets_data_from_lpinfo_multiscale(lpinfolist, stdsize=(32, 14), sizeratio=(1.,1.,1.)):
    datalist = load_lpcrdata_from_lpinfo_multiscale_pos_neg(lpinfolist, stdsize)
    train_set, valid_set, test_set = get_3sets_data_noshared(datalist, sizeratio)
    return train_set, valid_set, test_set


def get_3sets_data_from_lpinfo(lpinfolist, stdsize=(32, 14), sizeratio=(1.,1.,1.)):
    datalist = load_lpcrdata_from_lpinfo_pos_neg(lpinfolist, stdsize)
    train_set, valid_set, test_set = get_3sets_data(datalist, sizeratio)
#    train_set, valid_set, test_set = get_3sets_data_noshared(datalist, sizeratio)
    return train_set, valid_set, test_set


def load_lpcrdata_from_lpinfo_pos_neg_2(lpinfolist, stdsize=(32, 14)):
    datalist = []
    imgvecall = []
    tagall = []
    stdhalfw = stdsize[1] / 2
    for lp in lpinfolist:
        gimg = lp.charobj.grayimg
        schain = lp.charobj.state_chain
        
        imgw = gimg.shape[1]
        for wi in xrange(imgw-stdsize[1]):
            imgpart = gimg[:stdsize[0], wi:wi+stdsize[1]]
            imgvec = np.reshape(imgpart, stdsize[0]*stdsize[1])
            fimgvec = imgvec.astype(np.float32)
            fimgvec = normalize_data(fimgvec)
            imgvecall.append(fimgvec)
            
            if schain[wi+stdhalfw] > 0:
                tag = 1
            else:
                tag = 0
            tagall.append(tag)
#                imgvec = fimgvec * 255
#                imgvec = imgvec.astype(np.uint8);
#                cv2.imshow('rsz', imgvec.reshape(stdsize))
#                cv2.waitKey(0)
            
    imgvecall = np.asarray(imgvecall, dtype=np.float32)
    tagall = np.asarray(tagall, dtype=np.int32)
    print imgvecall.shape
    datalist.append((imgvecall, tagall))
    
    return datalist


def chknegpostag(bbs, chbbs):
    #1 is pos , 0 is neg, -1 is illegal
    bpos = 0
    billegal = 0
    h1 = bbs[3] - bbs[1]
    x1 = (bbs[0] + bbs[2]) / 2
    y1 = (bbs[1] + bbs[3]) / 2
    for bb in chbbs:
        x2 = (bb[0] + bb[2]) / 2
        y2 = (bb[1] + bb[3]) / 2
        h2 = bb[3] - bb[1]
        w2 = bb[2] - bb[0]
        distX = abs(x1 - x2)
        distY = abs(y1 - y2)
        distH = (h1 - h2)
        if distX * 4 <= w2 and distY == 0 and distH >= 0 and distH * 6 <= h2:
            bpos = 1
            break
        elif distX * 3 <= w2 and distY * 3 <= h2 and distH * 3 <= h2 :
            billegal = 1
            break
    
    tag = 0
    if bpos == 1:
        tag = 1
    elif billegal == 1:
        tag = -1
        
    return tag


def getsamples(lpimg, lpbbs, chbbs, stdsize):
    neglist = []
    taglist = []
    imgh, imgw = lpimg.shape
    stdh, stdw = stdsize
    negstepw = imgw / 20 #horizontal sample step
    negsteph = imgh / 10 #vertical sample step
    if imgh < stdh or imgw < stdw:
        return [], []
    mark = np.zeros_like(lpimg)
    for ri in xrange(0, imgh-stdh, 2):
        for ci in xrange(0, imgw-stdw, 2):
            imgpart = lpimg[ri:ri+stdh, ci:ci+stdw]
            tag = chknegpostag([ci, ri, ci + stdw, ri + stdh], chbbs)
            if tag != -1:
                if tag == 0:
                    if ri%negsteph != 0 or ci%negstepw != 0 or np.random.random() < 0.8: #accept ratio for neg samples
                        continue
                imgvec = np.reshape(imgpart, stdsize[0]*stdsize[1])
                fimgvec = imgvec.astype(np.float32)
                fimgvec = normalize_data(fimgvec)
                neglist.append(fimgvec)
                taglist.append(tag)
                
            if 0:
                if tag == 0:
                    mark[ri+stdh/2, ci+stdw/2] = 255
    if 0:
        allimg = mark / 2 + lpimg / 2
        cv2.imshow('mark', allimg)
        cv2.waitKey(0)
        
    return neglist, taglist


def getsamples_scales(lpimg, lpbbs, chbbs, stdsize):
    neglist = []
    taglist = []
    
    scales = np.asarray(range(50, 2, -1), dtype=np.float32)
    scales /= 50.0
    lpbbs = np.asarray(lpbbs)
    chbbs = np.asarray(chbbs)
    
    
    stdh, stdw = stdsize
    for rszv in scales:
        lpbbs_rsz = lpbbs * rszv
        lpbbs_rsz = lpbbs_rsz.astype(dtype=np.int32)
        chbbs_rsz = chbbs * rszv
        chbbs_rsz = chbbs_rsz.astype(dtype=np.int32)
        newrsz = (np.int(lpimg.shape[1] * rszv), np.int(lpimg.shape[0] * rszv))
        lpimg_rsz = cv2.resize(lpimg, newrsz)
#        print 'lp height:', lpbbs_rsz[3] - lpbbs_rsz[1], chbbs_rsz[0][3] - chbbs_rsz[0][1], lpimg.shape, 'std height:', stdsize[0]
        imgh, imgw = lpimg.shape
        if imgh < stdh or imgw < stdw:
            break
        neglistone, taglistone = getsamples(lpimg_rsz, lpbbs_rsz, chbbs_rsz, stdsize)
        
        neglist += neglistone
        taglist += taglistone
        
    return neglist, taglist



def load_lpcrdata_from_lpinfo_multiscale_pos_neg(lpinfolist, stdsize=(32, 14)):
    datalist = []
    imgvecall = []
    tagall = []
    for idx, lp in enumerate(lpinfolist):
        imgfn = lp.img_fn
        gimg = cv2.imread(imgfn, cv2.CV_LOAD_IMAGE_GRAYSCALE)
#        print gimg.flags
        chbbs = lp.char_bndbox
        bbs = lp.lp_bndbox
        margin = (bbs[3]-bbs[1]) / 2;
#        print bbs, gimg.shape
        bbs2 = [np.max([0, bbs[0]-margin]), np.max([0, bbs[1]-margin]), 
                np.min([gimg.shape[1]-1, bbs[2]+margin]), np.min([gimg.shape[0]-1, bbs[3]+margin])]
#        print chbbs[0][:2], chbbs[-1][2:]
        chrns = lp.char_name
        newchbbs = []
        lpimg = np.copy(gimg[bbs2[1]:bbs2[3]+1, bbs2[0]:bbs2[2]+1])
        newbbs = [bbs[0]-bbs2[0], bbs[1]-bbs2[1], bbs[2]-bbs2[0], bbs[3]-bbs2[1]]
        for i in xrange(len(chbbs)):
            bb = []
            ch = chrns[i]
            if ch == '-':
                continue
            newchbbs.append([chbbs[i][0] - bbs2[0], chbbs[i][1] - bbs2[1], chbbs[i][2] - bbs2[0], chbbs[i][3] - bbs2[1]])
        
        
#        print newchbbs
#        print newbbs
        
        if 0:
            clpimg = np.zeros((lpimg.shape[0], lpimg.shape[1], 3), dtype=np.uint8)
            clpimg[:, :, 0] = lpimg
            clpimg[:, :, 1] = lpimg
            clpimg[:, :, 2] = lpimg
            cv2.rectangle(clpimg, (newbbs[0], newbbs[1]), (newbbs[2], newbbs[3]), [0, 255, 0])
            for bb in newchbbs:
                cv2.rectangle(clpimg, (bb[0], bb[1]), (bb[2], bb[3]), [0, 0, 255])
            cv2.imshow('hi', clpimg)
            cv2.waitKey(40)
        
        
        if 0:
            veclist, taglist = getsamples_scales(lpimg, newbbs, newchbbs, stdsize)
        else:
            veclist, taglist = getsamples_scales_c(lpimg, newbbs, newchbbs, stdsize, 1000)
#        print lpimg.flags
#        cv2.imshow('lpimg', lpimg)
#        cv2.waitKey(0)
#        for fimgvec, tag in zip(veclist, taglist):
#            print tag
#            imgvec = np.asarray(fimgvec)
#            imgvec *= 255
#            imgvec = imgvec.astype(np.uint8)
#            imgvec = imgvec.reshape(stdsize)
#            cv2.imshow('rsz', imgvec)
#            cv2.waitKey(0)
            
#        taglistarray = np.asarray(taglist)
#        posnum = np.sum(taglistarray==1)
#        negnum = np.sum(taglistarray==0)
#        print idx, 'sample number:+%d -%d'%(posnum, negnum)
        imgvecall += veclist
        tagall += taglist
        
    
    imgvecall = np.asarray(imgvecall, dtype=np.float32)
    tagall = np.asarray(tagall, dtype=np.int32)
#    print 'vector shape:', imgvecall.shape
    datalist.append((imgvecall, tagall))
    
    return datalist
    

def load_lpcrdata_from_lpinfo_pos_neg(lpinfolist, stdsize=(32, 14)):
    datalist = []
    imgvecall = []
    tagall = []
    stdareasize = np.prod(stdsize)
    for lp in lpinfolist:
        gimg = lp.charobj.grayimg
        chkimg = np.zeros_like(gimg)
        bbs = lp.charobj.charbbs
        chrns = lp.char_name
        mark = np.zeros_like(gimg, dtype=np.int32)
        markvalue = 1
        for bb, chrn in zip(bbs, chrns):
            if chrn == '-':
                continue
            
            mark[bb[1]:bb[3]+1, bb[0]:bb[2]+1] = markvalue
            markvalue += 1
        
        
        imgw = gimg.shape[1]
        for wi in xrange(imgw-stdsize[1]):
            markpart = mark[:stdsize[0], wi:wi+stdsize[1]]
            sumarray = np.zeros(markvalue)
            for mv in xrange(markvalue):
                sumarray[mv] = np.sum(markpart==mv)
            
            maxi = np.argmax(sumarray)
            maxv = sumarray[maxi]
            
            imgpart = gimg[:stdsize[0], wi:wi+stdsize[1]]
            imgvec = np.reshape(imgpart, stdsize[0]*stdsize[1])
            fimgvec = imgvec.astype(np.float32)
            fimgvec = normalize_data(fimgvec)
            
            tag = -1
            if maxi > 0:
                if maxv * 100 > stdareasize * 80:
                    imgvecall.append(fimgvec)
                    tag = 1
                    tagall.append(tag)
                elif maxv * 100 < stdareasize * 60:
                    imgvecall.append(fimgvec)
                    tag = 0
                    tagall.append(tag)
            elif maxi == 0: # and maxv * 100 > stdareasize * 40:
                imgvecall.append(fimgvec)
                tag = 0
                tagall.append(tag)
            
            if 0 and tag == 1:
#                print sumarray, maxi, maxv, stdareasize
                chkimg[:, wi+stdsize[1]/2] = 255
#                imgvec = fimgvec * 255
#                imgvec = imgvec.astype(np.uint8);
#                cv2.imshow('rsz', imgvec.reshape(stdsize))
#                cv2.waitKey(0)
        if 0:
            spimg = chkimg / 2 + gimg / 2
            mark2 = np.zeros_like(mark, dtype=np.uint8)
            mark2[mark==6] = 255
            spimg2 = gimg / 2 + mark2 / 2
            allimg = np.append(spimg2, spimg, axis=0)
            cv2.imshow('spimg', allimg)
            cv2.waitKey(0)
    
    imgvecall = np.asarray(imgvecall, dtype=np.float32)
    tagall = np.asarray(tagall, dtype=np.int32)
    print imgvecall.shape
    datalist.append((imgvecall, tagall))
    
    return datalist


def load_lpcrdata_from_lpinfo_pos(lpinfolist, stdsize=(32, 14)):
    datalist = []
    imgvecall = []
    tagall = []
    for lp in lpinfolist:
        gimg = lp.charobj.grayimg
        schain = lp.charobj.state_chain
        bbs = lp.charobj.charbbs
        chrns = lp.char_name
        for bb, chrn in zip(bbs, chrns):
            if chrn == '-':
                continue
            imgpart = gimg[bb[1]:bb[3]+1, bb[0]:bb[2]+1]
            imgrsz = cv2.resize(imgpart, (stdsize[1], stdsize[0])) #resize image to 28x28
#            cv2.imshow('char', imgrsz)
            imgvec = np.reshape(imgrsz, stdsize[0]*stdsize[1])
            fimgvec = imgvec.astype(np.float32)
#            lpcr_func_c.normalize_img_data_to_0_1(imgvec.ctypes, fimgvec.ctypes, stdsize[0]*stdsize[1], 20);
            fimgvec = normalize_data(fimgvec)
            imgvecall.append(fimgvec)
            tagall.append(1)
            
#            imgvec = fimgvec * 255
#            imgvec = imgvec.astype(np.uint8);
#            cv2.imshow('rsz', imgvec.reshape(stdsize))
#            cv2.waitKey(0)
            
    imgvecall = np.asarray(imgvecall, dtype=np.float32)
    tagall = np.asarray(tagall, dtype=np.int32)
    
#    print imgvecall.shape
    datalist.append((imgvecall, tagall))
    
    return datalist
    

def load_lpcrdata_from_image(folder_name,stdsize=(28,28)):
# return: all data
#    print folder_name
    files=os.listdir(folder_name)
#    print len(files)
    datalist=list()
    for fname in files:
        fullpath=folder_name+'/'+fname
        if os.path.isdir(fullpath):
            files2=os.listdir(fullpath)
            num2=len(files2)
            imgvecall=np.zeros((num2,stdsize[0]*stdsize[1]),dtype=np.float32)
            tagall=np.zeros(num2,dtype=np.int32)
            imgi=0
            for imgname in files2:
                imgpath=fullpath+'/'+imgname
#                print imgpath
                img=cv2.imread(imgpath)
                img=img[:,:,0]
                imgrsz=cv2.resize(img,stdsize) #resize image to 28x28
                imgvec=np.reshape(imgrsz,stdsize[0]*stdsize[1])
                fimgvec=imgvec.astype(np.float32)
#                imgvec=normalize_data(imgvec)
#                imgvec=normalize_img_data_to_0_1(imgvec,10)
                lpcr_func_c.normalize_img_data_to_0_1(imgvec.ctypes, fimgvec.ctypes, stdsize[0]*stdsize[1], 20)
#                fimgvec = normalize_img_data_to_unitvar(imgvec)
#                fimgvec -= np.mean(fimgvec)
#                print fimgvec
                imgvecall[imgi,:]=fimgvec
                tagall[imgi]=np.int(fname)
                imgi+=1
                
#                imgvec=np.multiply(fimgvec,255)
#                imgvec=imgvec.astype(np.uint8);
#                cv2.imshow('rsz',imgvec.reshape((28,28)))
#                cv2.waitKey(0)
            datalist.append((imgvecall,tagall))
    return datalist


def normalize_img_data_to_unitvar(imgvec):
    fimgvec = imgvec.astype(np.float32)
    stdv = np.std(fimgvec)
    fimgvec /= stdv
    
    return fimgvec

def normalize_img_data_to_0_1_c(imgvec,ratio=20):
    fimgvec = np.zeros_like(imgvec, dtype=np.float32)
    lpcr_func_c.normalize_img_data_to_0_1(imgvec.ctypes, fimgvec.ctypes, imgvec.shape[0], ratio)
    return fimgvec
    
def normalize_img_data_to_0_1(imgvec,ratio=20):
#normalize image data into [0, 1]
#ratio: ratio of all pixels
    vlen=len(imgvec)
    hist=np.zeros(256,np.int32)
    for i in imgvec:
        hist[i]+=1
    
    mininfo=[0,0]
    maxinfo=[0,0]
    minv=np.amin(imgvec)
    maxv=np.amax(imgvec)
    for i in xrange(minv,maxv+1):
        mininfo[0]+=hist[i]
        mininfo[1]+=hist[i]*i
        if mininfo[0]*100>vlen*ratio:
            break
    
    for i in xrange(maxv,minv-1,-1):
        maxinfo[0]+=hist[i]
        maxinfo[1]+=hist[i]*i
        if maxinfo[0]*100>vlen*ratio:
            break
    minv=mininfo[1]/mininfo[0]
    maxv=maxinfo[1]/maxinfo[0]
#    print minv, maxv
    imgvec=imgvec.astype(np.float32)
    imgvec=(imgvec-minv)/(maxv-minv+0.000001)
    imgvec[imgvec<0]=0.
    imgvec[imgvec>1]=1.
    
    return imgvec

def normalize_data(imgvec):
    imgvec=imgvec.astype(np.float32)
    minv=np.amin(imgvec)
    maxv=np.amax(imgvec)
#                print minv, maxv
    imgvec=(imgvec-minv)/(maxv-minv+0.000001)
    return imgvec


def get_3sets_data_noshared(datalist,sizeratio=(1.,1.,1.)):
#traning set
#validate set
#test set
    sizeratio=np.asarray(sizeratio, dtype=np.float32)
    sizeratio=sizeratio/sum(sizeratio)
#    print sizeratio
    clsnum = len(datalist)
    num3eachlist = np.zeros((clsnum,3), dtype=np.int32)
    ci = 0
    for clsdata in datalist:
        onetag = clsdata[1]
        datanum = len(onetag)
        num3each = np.multiply(sizeratio, datanum)
        num3each = num3each.astype(np.int32)
#        print onetag[0],'(',datanum,')',':',num3each
        num3eachlist[ci,:] = num3each
        ci += 1
#    print num3eachlist
    
    datadim = np.shape(datalist[0][0])[1]
    num3all = np.sum(num3eachlist, 0);
#    print datadim,num3all
    train_set_x = np.zeros((num3all[0], datadim), dtype=np.float32)
    train_set_y = np.zeros(num3all[0], dtype = np.int32)
    valid_set_x = np.zeros((num3all[1], datadim), dtype=np.float32)
    valid_set_y = np.zeros(num3all[1], dtype=np.int32)
    test_set_x = np.zeros((num3all[2], datadim),dtype=np.float32)
    test_set_y = np.zeros(num3all[2], dtype=np.int32)
    ei = 0
    num3each_tmp = [0, 0, 0]
    for clsdata in datalist:
        onedata = clsdata[0]
        onetag = clsdata[1]
        
        #random permutation
        onedata, onetag = randomperm_data_tag(onedata, onetag)
#        print onedata.shape
        
        num3each=num3eachlist[ei]
        starti=0
        endi=num3each[0]
#        print num3each,valid_set_x.shape
        train_set_x[num3each_tmp[0]:num3each_tmp[0]+num3each[0],:] = onedata[starti:endi,:]
        train_set_y[num3each_tmp[0]:num3each_tmp[0]+num3each[0]] = onetag[starti:endi]
        
        starti=endi
        endi=starti+num3each[1]
        valid_set_x[num3each_tmp[1]:num3each_tmp[1]+num3each[1],:] = onedata[starti:endi,:]
        valid_set_y[num3each_tmp[1]:num3each_tmp[1]+num3each[1]] = onetag[starti:endi]
        
        starti = endi
        endi = starti+num3each[2]
        test_set_x[num3each_tmp[2]:num3each_tmp[2]+num3each[2],:] = onedata[starti:endi,:]
        test_set_y[num3each_tmp[2]:num3each_tmp[2]+num3each[2]] = onetag[starti:endi]
        
        ei += 1
        num3each_tmp += num3each
#        print num3each_tmp
#    print num3each_tmp
    if 0:
        print 'whitening...'
        U, S = whitening.get_U_S_image(train_set_x.T)
        train_set_x = whitening.zca_white_all(train_set_x.T, U, S).T
        valid_set_x = whitening.zca_white_all(valid_set_x.T, U, S).T
        test_set_x = whitening.zca_white_all(test_set_x.T, U, S).T
#        datalen = test_set_x.shape[0]
#        for ni in xrange(0, datalen, 1):
#            vec = test_set_x[ni, :]
#            img = whitening.vec_2_img(vec, (28, 28))
#            cv2.imshow('img', img)
#            cv2.waitKey(0)
    
    train_set_x, train_set_y = randomperm_data_tag(train_set_x, train_set_y)
    
    valid_set_x, valid_set_y = randomperm_data_tag(valid_set_x, valid_set_y)
    
    test_set_x, test_set_y = randomperm_data_tag(test_set_x, test_set_y)   
    
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)
    
    return train_set, valid_set, test_set
    

def get_3sets_data(datalist,sizeratio=(1.,1.,1.)):
#traning set
#validate set
#test set
    sizeratio=np.asarray(sizeratio, dtype=np.float32)
    sizeratio=sizeratio/sum(sizeratio)
#    print sizeratio
    clsnum = len(datalist)
    num3eachlist = np.zeros((clsnum,3), dtype=np.int32)
    ci = 0
    for clsdata in datalist:
        onetag = clsdata[1]
        datanum = len(onetag)
        num3each = np.multiply(sizeratio, datanum)
        num3each = num3each.astype(np.int32)
#        print onetag[0],'(',datanum,')',':',num3each
        num3eachlist[ci,:] = num3each
        ci += 1
#    print num3eachlist
    
    datadim = np.shape(datalist[0][0])[1]
    num3all = np.sum(num3eachlist, 0);
#    print datadim,num3all
    train_set_x = np.zeros((num3all[0], datadim), dtype=np.float32)
    train_set_y = np.zeros(num3all[0], dtype = np.int32)
    valid_set_x = np.zeros((num3all[1], datadim), dtype=np.float32)
    valid_set_y = np.zeros(num3all[1], dtype=np.int32)
    test_set_x = np.zeros((num3all[2], datadim),dtype=np.float32)
    test_set_y = np.zeros(num3all[2], dtype=np.int32)
    ei = 0
    num3each_tmp = [0, 0, 0]
    for clsdata in datalist:
        onedata = clsdata[0]
        onetag = clsdata[1]
        
        #random permutation
        onedata, onetag = randomperm_data_tag(onedata, onetag)
#        print onedata.shape
        
        num3each=num3eachlist[ei]
        starti=0
        endi=num3each[0]
#        print num3each,valid_set_x.shape
        train_set_x[num3each_tmp[0]:num3each_tmp[0]+num3each[0],:] = onedata[starti:endi,:]
        train_set_y[num3each_tmp[0]:num3each_tmp[0]+num3each[0]] = onetag[starti:endi]
        
        starti=endi
        endi=starti+num3each[1]
        valid_set_x[num3each_tmp[1]:num3each_tmp[1]+num3each[1],:] = onedata[starti:endi,:]
        valid_set_y[num3each_tmp[1]:num3each_tmp[1]+num3each[1]] = onetag[starti:endi]
        
        starti = endi
        endi = starti+num3each[2]
        test_set_x[num3each_tmp[2]:num3each_tmp[2]+num3each[2],:] = onedata[starti:endi,:]
        test_set_y[num3each_tmp[2]:num3each_tmp[2]+num3each[2]] = onetag[starti:endi]
        
        ei += 1
        num3each_tmp += num3each
#        print num3each_tmp
#    print num3each_tmp
    if 0:
        print 'whitening...'
        U, S = whitening.get_U_S_image(train_set_x.T)
        train_set_x = whitening.zca_white_all(train_set_x.T, U, S).T
        valid_set_x = whitening.zca_white_all(valid_set_x.T, U, S).T
        test_set_x = whitening.zca_white_all(test_set_x.T, U, S).T
#        datalen = test_set_x.shape[0]
#        for ni in xrange(0, datalen, 1):
#            vec = test_set_x[ni, :]
#            img = whitening.vec_2_img(vec, (28, 28))
#            cv2.imshow('img', img)
#            cv2.waitKey(0)
    
    train_set_x, train_set_y = randomperm_data_tag(train_set_x, train_set_y)
    train_set_x = shared(train_set_x)
    train_set_y = shared(train_set_y)
    
    valid_set_x, valid_set_y = randomperm_data_tag(valid_set_x, valid_set_y)
    valid_set_x = shared(valid_set_x)
    valid_set_y = shared(valid_set_y)
    
    test_set_x, test_set_y = randomperm_data_tag(test_set_x, test_set_y)
    test_set_x = shared(test_set_x)
    test_set_y = shared(test_set_y)    
    
    train_set = (train_set_x, train_set_y)
    valid_set = (valid_set_x, valid_set_y)
    test_set = (test_set_x, test_set_y)
    
    return train_set, valid_set, test_set


def randomperm_data_tag(data,tag):
    dlen = len(tag)
    didx = np.arange(dlen)
    np.random.shuffle(didx)
    data_o = np.zeros(data.shape,data.dtype)
    tag_o = np.zeros(tag.shape,tag.dtype)
    pi = 0
    for idx in didx:
        data_o[pi,:] = data[idx,:]
        tag_o[pi] = tag[idx]
        pi += 1
    return data_o, tag_o

def get_3sets_data_from_images(folder_name,stdsize=(28,28),sizeratio=(1.,1.,1.)):
    datalist=load_lpcrdata_from_image(folder_name,stdsize)
    train_set,valid_set,test_set=get_3sets_data(datalist,sizeratio)
   
    return train_set,valid_set,test_set

def get_tensor_data(tensordata):
    datafunc=theano.function([],outputs=tensordata)
    return datafunc()

def set_tensor_data(npdata,tensordata):
    datafunc=theano.function([],[],updates=[(tensordata,npdata)])
    return datafunc()
#get_3sets_data_from_images('/media/mzhang/data/twcharset0_9',stdsize=(28,28),sizeratio=(2.,1.,1.))

#myfile = open('param/cnn.params0.3478','r')
#param = cPickle.load(myfile)
#print param
##showdata.showTensorData(param[4])
#myfile.close()

#import lpfunctions as lpfuncs
#folderpath = '/Volumes/ZMData1/LPR_TrainData/new/'
#stdsize=(28, 14)
#lpinfo_list, whist = lpfuncs.getall_lps2(folderpath, 20, stdsize[0], ifstrech=False)
#load_lpcrdata_from_lpinfo_multiscale_pos_neg(lpinfo_list, stdsize=stdsize)







