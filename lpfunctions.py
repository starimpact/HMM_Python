# -*- coding: utf-8 -*-
"""
Created on Thu Oct 30 14:18:07 2014

@author: mzhang
"""

import xml.etree.ElementTree as ele
import os
import numpy as np
import cv2 as cv
import functions as funcs


class CHR_Info:
    def __init__(self):
        self.grayimg = None #gray image
        self.sblimg = None #sobel image
        self.grayimgf = None #float gray image
        self.sblimgf = None #float sobel image
        self.mskimg = None #mask image
        self.state_chain = None #state chain
        self.obs_chain = None #observation chain
        self.charbbs = [] #char's bound box
        self.lp_bndbox = []


class LP_Info:
    def __init__(self):
        self.xml_fn = ''
        self.img_fn = ''
        self.lp_bndbox = [0, 0, 0, 0]
        self.char_num = 0
        self.char_bndbox = []
        self.char_name = ''
        self.charobj = CHR_Info()
    
    

def getnum(txt):
    val = int(txt.text)
    return val
    
    
def getbndbox(bndbox):
    xmin = getnum(bndbox.find('xmin'))
    ymin = getnum(bndbox.find('ymin'))
    xmax = getnum(bndbox.find('xmax'))
    ymax = getnum(bndbox.find('ymax'))
    
    return [xmin, ymin, xmax, ymax]
    

def readlp(fn, lpinfo):
    f = open(fn, 'r')
    xmlcont = f.read()
    f.close()
    xmlcont = xmlcont.lower().replace('gb2312', 'utf-8')
    root = ele.fromstring(xmlcont)
    obj = root.find('object')
    bndbox = obj.find('bndbox')
    lpbndbox = getbndbox(bndbox)
    
    parts = obj.findall('part')
    partsbnd = []
    charsname = ''
    for part in parts:
        bndbox = part.find('bndbox')
        bbval = getbndbox(bndbox)
        partsbnd.append(bbval)
        
        chrname = part.find('name')
        charsname += chrname.text
    
    lpinfo.char_num = len(parts)
    lpinfo.lp_bndbox = lpbndbox
    lpinfo.char_bndbox = partsbnd
    lpinfo.char_name = charsname
    
#    return lpinfo

def readall(folderpath):
    lpinfo_list = []
    objs = os.listdir(folderpath)
    for obj in objs:
        onefn = folderpath + obj
        if os.path.isfile(onefn) == False:
            continue
        fn, ext = os.path.splitext(obj)
        if ext == '.xml':
            lpinfo = LP_Info()
            lpinfo.xml_fn = folderpath + obj
            lpinfo.img_fn = folderpath + fn + '.jpg'
            readlp(onefn, lpinfo)
#            print lpinfo.char_num
            lpinfo_list.append(lpinfo)
            
#            print lpinfo.lp_bndbox
#            print lpinfo.char_num
#            print lpinfo.char_bndbox
    return lpinfo_list


def readall2(folderpath, neednum):
    lpinfo_list = []
    xmlfd = folderpath + 'XML/'
    imgfd = folderpath + 'IMG/'
    objs = os.listdir(xmlfd)
    nownum = 0
    for obj in objs:
#        onefn = folderpath + obj
#        print onefn
#        if os.path.isfile(onefn) == False:
#            continue
        
        fn, ext = os.path.splitext(obj)
        if ext == '.xml':
            lpinfo = LP_Info()
            lpinfo.xml_fn = xmlfd + obj
            lpinfo.img_fn = imgfd + fn + '.jpg'
            readlp(lpinfo.xml_fn, lpinfo)
#            print lpinfo.char_num
            lpinfo_list.append(lpinfo)
            
            nownum += 1
            if neednum >= 0 and nownum >= neednum:
                break
#            print lpinfo.lp_bndbox
#            print lpinfo.char_num
#            print lpinfo.char_bndbox
    
    return lpinfo_list


def showbndbox(lpinfo):
    import cv2 as cv
#    img = cv.imread(lpinfo.img_fn, cv.CV_LOAD_IMAGE_GRAYSCALE)
    img = cv.imread(lpinfo.img_fn)
    bb = lpinfo.lp_bndbox
    mark = img.copy()
    cv.rectangle(mark, bb[:2], bb[2:], (0, 0, 255), 2)
    marksz = cv.resize(mark, (mark.shape[1] / 2, mark.shape[0] / 2))
    cv.imshow('img', marksz)
    cv.waitKey(0)


def hist_strech(img):
    hist, edges = np.histogram(img, bins=np.arange(0, 257))
    ratio = 0.4
    val = img.shape[0] * img.shape[1] * ratio
    
    meanv = 0
    nbr = 0
    for i in xrange(256):
        nbr += hist[i]
        meanv += hist[i] * i
        if nbr >= val:
            break
    lv = meanv / nbr
    
    meanv = 0
    nbr = 0
    for i in xrange(255, -1, -1):
        nbr += hist[i]
        meanv += hist[i] * i
        if nbr >= val:
            break
    hv = meanv / nbr
    
    dif = hv - lv + 1
    dif = 1 if dif < 1 else dif
#    print lv, hv, dif
    strechimg = np.zeros_like(img)
    for i in xrange(img.shape[0]):
        for j in xrange(img.shape[1]):
            tmp = img[i, j] - lv
            tmp = 0 if tmp < 0 else tmp
            tmp = tmp * 255 / dif
            tmp = 255 if tmp > 255 else tmp
            strechimg[i, j] = tmp
    
    return strechimg


def calc_newbb(bbold, bboffset):
    nbb = [0, 0, 0, 0]
    nbb[0] = bbold[0] - bboffset[0]
    nbb[1] = bbold[1] - bboffset[1]
    nbb[2] = bbold[2] - bboffset[0]
    nbb[3] = bbold[3] - bboffset[1]
    
    return nbb
    

def showcharbb(charbb, imggray):
    cimg = cv.cvtColor(imggray, cv.COLOR_GRAY2BGR)
    for bb in charbb:
        cv.rectangle(cimg, tuple(bb[:2]), tuple(bb[2:]), (0, 0, 255), 1)
    cv.imshow('charbb', cimg)
    

def generateMask(grayimg, chars_name, chars_bb):
    mask = np.zeros_like(grayimg)
    mkn = 0
    for nm, bb in zip(chars_name, chars_bb):
        if nm == '-':
            continue
        print '\'%s\':%d'%(nm, bb[2] - bb[0]), 
        mkn += 1
        mask[bb[1]:bb[3]+1, bb[0]:bb[2]+1] = mkn
    print
    
    return mask


def get_state_chain(mask_chain):
    state_chain = np.zeros_like(mask_chain)
    chain_len = mask_chain.shape[0]
    for i in xrange(1, chain_len):
        msk_pre = mask_chain[i-1]
        msk_now = mask_chain[i]
        if msk_now > 0:
            if msk_now != msk_pre:
                state_chain[i] = 1
            else:
                state_chain[i] = state_chain[i-1] + 1
    return state_chain


def extract_lpinfo(lpinfo, rszh = 32, ifstrech=True):
    img = cv.imread(lpinfo.img_fn, cv.CV_LOAD_IMAGE_GRAYSCALE)
#    img = cv.imread(lpinfo.img_fn)
    bb = np.copy(lpinfo.lp_bndbox)
    
    extendy = (bb[3] - bb[1]) / 10 #np.random.randint(2, 4)
    extendx = (bb[2] - bb[0]) / 4 #np.random.randint(16, 32)
    
    
    bb[0] = 0 if bb[0] - extendx < 0 else bb[0] - extendx
    bb[1] = 0 if bb[1] - extendy < 0 else bb[1] - extendy
    bb[2] = img.shape[1]-1 if bb[2] + extendx >= img.shape[1] else bb[2] + extendx
    bb[3] = img.shape[0]-1 if bb[3] + extendy >= img.shape[0] else bb[3] + extendy
        
    
    lpimg = img[bb[1]:bb[3], bb[0]:bb[2]]
    #image enhancement by histogram streching
    if ifstrech:
        strechimg = hist_strech(lpimg)
    else:
        strechimg = lpimg
    
    #resize image into the uniform height
    resizehight = rszh
    resizerate = resizehight * 1.0 / strechimg.shape[0]
    newshape = [0, 0]
    newshape[0] = resizehight
    newshape[1] = np.int(strechimg.shape[1] * resizerate)
    strechimg = cv.resize(strechimg, (newshape[1], newshape[0]))
    #avoid zero column
    tmpgray = strechimg.astype(np.int32) # + np.random.randint(-8, 9, strechimg.shape)
    tmpgray[tmpgray < 0] = 0
    tmpgray[tmpgray > 255] = 255
    strechimg = tmpgray.astype(np.uint8)
    
    sblimg = funcs.siSobel_U8(strechimg)
#    zero_col_mark = np.sum(sblimg, axis=0)==0
#    sblimg[:, zero_col_mark] = np.random.randint(0, 4, (resizehight, np.sum(zero_col_mark)))
#    sblimg = funcs.siSobelX_U8(strechimg)
    
    if 0:
        allimg = np.append(strechimg, sblimg, axis=1)
        cv.imshow('img', allimg)
        cv.waitKey(10)
    
    #calc new bound box
    lpinfo.charobj.lp_bndbox = calc_newbb(lpinfo.lp_bndbox, bb)
    lpinfo.charobj.grayimg = np.copy(strechimg)
    lpinfo.charobj.sblimg = np.copy(sblimg)
#    lpinfo.charobj.grayimgf = strechimg.astype(np.float32) / 255
#    lpinfo.charobj.sblimgf = sblimg.astype(np.float32) / 255
    lpinfo.charobj.charbbs = []
    for i, bbone in enumerate(lpinfo.char_bndbox):
        bbnew = calc_newbb(bbone, bb)
        for i in xrange(len(bbnew)):
            bbnew[i] = np.int(bbnew[i] * resizerate + 0.5)
        lpinfo.charobj.charbbs.append(bbnew)
    
    #show the char boundbox of LP
    showcharbb(lpinfo.charobj.charbbs, lpinfo.charobj.grayimg)
    print lpinfo.char_name, ', [', extendx, ', ', extendy, ']'
    
    #get mask image
    mask = generateMask(lpinfo.charobj.grayimg, lpinfo.char_name, lpinfo.charobj.charbbs)
    lpinfo.charobj.mskimg = mask
    #get state chain from middle row of mask image
    mask_chain = mask[mask.shape[0] / 2, :]
    state_chain = get_state_chain(mask_chain)
    lpinfo.charobj.state_chain = np.asarray(state_chain, dtype=np.int32)
#    print state_chain
    #get observation chain
    obs_chain = np.sum(lpinfo.charobj.sblimg, axis=0) / lpinfo.charobj.sblimg.shape[0]
    obs_chain /= 2
    lpinfo.charobj.obs_chain = np.asarray(obs_chain, dtype=np.int32)
#    print obs_chain
    
    


def row_normalize(data):
    ndata = np.zeros_like(data)
    for i in xrange(data.shape[0]):
        row = data[i, :]
        row_sum = np.sum(row)
        ndata[i, :] = row / row_sum
    
    return ndata


def getall_lps(folderpath, neednum=-1, ifstrech=True):
    lpinfo_list = readall(folderpath)
    lpinfo_num = len(lpinfo_list)
    
    if neednum < 0 or neednum > lpinfo_num:
        neednum = lpinfo_num
    
    print neednum
    
    for i in xrange(neednum):
        lpi = lpinfo_list[i]
        print '%d:%s'%(i, lpi.img_fn)
        extract_lpinfo(lpi, ifstrech)
        print '-----------------------------'
    lpinfo_list = lpinfo_list[:neednum]
    
    return lpinfo_list


def getall_lps2(folderpath, neednum=-1, rszh = 32, ifstrech=True):
    lpinfo_list = readall2(folderpath, neednum)
    lpinfo_num = len(lpinfo_list)
    print lpinfo_num
    if neednum < 0 or neednum > lpinfo_num:
        neednum = lpinfo_num
    
    print neednum
    hist = {}
    for i in xrange(neednum):
        lpi = lpinfo_list[i]
        print '%d:%s'%(i, lpi.img_fn)
        extract_lpinfo(lpi, rszh, ifstrech)
        bbs = lpi.charobj.charbbs
        chname = lpi.char_name
        for j in xrange(len(bbs)):
            if chname[j] == '-':
                continue
            bb = bbs[j]
            cw = bb[2]-bb[0]
            if hist.has_key(cw) == False:
                hist[cw] = 1
            else:
                hist[cw] += 1
        
        print '-----------------------------'
    lpinfo_list = lpinfo_list[:neednum]
#    print hist
    
    return lpinfo_list, hist
    
