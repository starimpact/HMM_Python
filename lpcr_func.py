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

lpcr_func_c = ctypes.CDLL('./lpcr_func_c.so')


def get_3sets_data_from_lpinfo(lpinfolist, stdsize=(32, 14), sizeratio=(1.,1.,1.)):
    datalist = load_lpcrdata_from_lpinfo_pos_neg(lpinfolist, stdsize)
    train_set, valid_set, test_set = get_3sets_data(datalist, sizeratio)
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
    
    

def load_lpcrdata_from_lpinfo_pos_neg(lpinfolist, stdsize=(32, 14)):
    datalist = []
    imgvecall = []
    tagall = []
    stdareasize = np.prod(stdsize)
    for lp in lpinfolist:
        gimg = lp.charobj.grayimg
        bbs = lp.charobj.charbbs
        chrns = lp.char_name
        mark = np.zeros_like(gimg, dtype=np.int32)
        markvalue = 1
        for bb, chrn in zip(bbs, chrns):
            if chrn == '-':
                continue
            
            mark[bb[1]:bb[3]+1, bb[0]:bb[2]+1] = markvalue
            markvalue += 1
        
        sumarray = np.zeros(markvalue)
        imgw = gimg.shape[1]
        for wi in xrange(imgw-stdsize[1]):
            markpart = mark[:stdsize[0], wi:wi+stdsize[1]]
            
            for mv in xrange(markvalue):
                sumarray[mv] = np.sum(markpart==mv)
            
            maxi = np.argmax(sumarray)
            maxv = sumarray[maxi]
#            print sumarray, maxi, maxv
            
            imgpart = gimg[:stdsize[0], wi:wi+stdsize[1]]
            imgvec = np.reshape(imgpart, stdsize[0]*stdsize[1])
            fimgvec = imgvec.astype(np.float32)
            fimgvec = normalize_data(fimgvec)
            
            if maxi > 0:
                if maxv * 100 > stdareasize * 70:
                    imgvecall.append(fimgvec)
                    tag = 1
                    tagall.append(tag)
                elif maxv * 100 < stdareasize * 60:
                    imgvecall.append(fimgvec)
                    tag = 0
                    tagall.append(tag)
            elif maxi == 0 and maxv * 100 > stdareasize * 40:
                imgvecall.append(fimgvec)
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
                lpcr_func_c.normalize_img_data_to_0_1(imgvec.ctypes, fimgvec.ctypes, stdsize[0]*stdsize[1], 20);
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






