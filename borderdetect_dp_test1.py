# -*- coding: utf-8 -*-
"""
Created on Sat Jan  3 19:35:16 2015

@author: mzhang
"""


import cv2 as cv
import numpy as np
import functions as funcs
import matplotlib.pyplot as pp


def siPrintArray2D(fmt, aData):
    if len(aData.shape) != 2:
        print 'only support 2D array.'
        return
    h, w = aData.shape
    print 'Data:'
    print '%3c:['%(' '), 
    for j in xrange(w):
        print fmt%(j), 
    print ']'
    for i in xrange(h):
        print '%3d:['%(i), 
        for j in xrange(w):
            print fmt%(aData[i, j]), 
        print ']'
        

def get_line_score(segimg, x1, x2):
    imgh, imgw = segimg.shape
    score = 0
    invh = 1.0 / imgh
    for hi in xrange(imgh):
        imgrow = segimg[hi, :]
        xi = x1 + hi * (x2 - x1) * invh
        lxi = np.int(xi)
        r = xi - lxi
        hxi = lxi + 1
        vll = imgrow[lxi-1]
        vl = imgrow[lxi]
        vr = imgrow[hxi]
        vrr = imgrow[hxi+1]
        v = vl * (1 - r) + vr * r
        score += v
#        score += vll + vl + vr + vrr
    
    avgscore = score
    avgscore = score*10/imgh
#    if avgscore < 10:
#        avgscore = 0.
    return avgscore
    
    
def generateGraph(gimg):
    imgh, imgw = gimg.shape
    graph = np.zeros((imgw, imgw), dtype=np.int32)
    radius = imgh / 10
    print radius
    for x1 in xrange(4, imgw-4):
        for x2 in xrange(x1-radius, x1+radius+1):
            if x2 < 1 or x2 >= imgw-1:
                continue
            graph[x1, x2] = get_line_score(gimg, x1, x2)
    
    return graph


def findLinesByDP(graph):
    print 'graph shape:', graph.shape
    imgh, imgw = graph.shape
    if imgh != imgw:
        print 'findLinesByDP error....'
        exit()
    radius = 2
    pathgraph = np.zeros_like(graph, dtype=np.int32)
    maxgraph = np.copy(graph)
    for ri in xrange(1, imgh):
        rowmaxpre = maxgraph[ri-1, :]
        rowmaxnow = maxgraph[ri, :]
        pathrownow = pathgraph[ri, :]
        for ci in xrange(radius, imgw):
#            rowseg = rowmaxpre[:ci]
#            rowseg = rowmaxpre[:ci+1]
            rowseg = rowmaxpre[ci-radius:ci]
            pathrownow[ci] = np.argmax(rowseg) - radius
            rowmaxnow[ci] += np.max(rowseg)
#        print pathrownow
    
#    siPrintArray2D('%3d', pathgraph)
#    siPrintArray2D('%4d', maxgraph)
    xx = []
    lastrow = imgh - 1
    rowmaxlast = maxgraph[lastrow, :]
    
    rowmaxpathi = np.argmax(rowmaxlast)
    lastcol = rowmaxpathi
    colmaxlast = maxgraph[:, lastcol]
    colmaxpathi = np.argmax(colmaxlast)
    
    print 'max:%d,%d\n'%(colmaxpathi, rowmaxpathi)
    
    maxpathpre_subidx = pathgraph[colmaxpathi, rowmaxpathi]
    xxone = [colmaxpathi, rowmaxpathi]
    print xxone
    while(1):
        xxone[0] -= 1
        xxone[1] += maxpathpre_subidx
        if graph[xxone[0], xxone[1]] > 0:
            xx.append(np.copy(xxone))
        if xxone[0] == 0 or xxone[1] == 0:
            break
        maxpathpre_subidx = pathgraph[xxone[0], xxone[1]]
#        print xxone[0], maxpathi
    xx = np.asarray(xx)
    return xx


def filterRepeatLine(graph, xx):
    graphnew = np.zeros_like(graph)
    xnum = xx.shape[0]
    for i in xrange(xnum):
        x = xx[i, :]
        graphnew[x[0], x[1]] = graph[x[0], x[1]]
    imgw, imgw = graphnew.shape
    xxnew = []
    for i in xrange(imgw):
        col = graphnew[:, i]
        maxv = np.max(col)
        maxi = np.argmax(col)
        if maxv > 0:
            tmp = [maxi, i]
            xxnew.append(tmp)
    xxnew = np.asarray(xxnew)
    return xxnew
        


def showBiMaxConnection(img, gimg, graph, xx):
    imgh, imgw = gimg.shape
    print xx.shape
    pnum = xx.shape[0]
    zoom = 2
    zgimg = cv.resize(img, (np.int(imgw*zoom), imgh))
    imgh, imgw = zgimg.shape
    cimg = cv.cvtColor(zgimg, cv.COLOR_GRAY2BGR)
    cmrk = np.zeros_like(cimg)
    zxx = (xx * zoom + 0.5)
    zxx = zxx.astype(np.int32)
    for pi in xrange(pnum):
        x1 = xx[pi, 0]
        x2 = xx[pi, 1]
#        print graph[x1, x2]
        xs = zxx[pi, :]
        cv.line(cimg, (xs[0], 0), (xs[1], imgh-1), (0, 0, 255), 1)
#    fusionimg = cimg * 2 / 3 + cmrk / 3
#    allimg = np.append(cimg, fusionimg, axis=0)
#    cv.imshow('markline_max', cimg)
#    cv.waitKey(10)
    return cimg


def doNMS(graph):
    gh, gw = graph.shape
    nmsgraph = np.zeros_like(graph)
    for hi in xrange(1, gh-1):
        for wi in xrange(1, gw-1):
            if graph[hi, wi] > graph[hi-1, wi-1] \
               and graph[hi, wi] > graph[hi+1, wi+1] \
               and graph[hi, wi] > graph[hi+1, wi] \
               and graph[hi, wi] > graph[hi-1, wi] \
               and graph[hi, wi] > graph[hi, wi+1] \
               and graph[hi, wi] > graph[hi, wi-1] \
               and graph[hi, wi] > graph[hi-1, wi+1] \
               and graph[hi, wi] > graph[hi+1, wi-1]:
                   nmsgraph[hi, wi] = graph[hi, wi]
#                   nmsgraph[hi, wi] = 1
    return nmsgraph
    


def get_edge(gradimg):
    imgh, imgw = gradimg.shape
    edgeimg = np.zeros_like(gradimg)
    for ri in xrange(imgh):
        for ci in xrange(1, imgw-1):
            if gradimg[ri, ci] > 10 and \
                gradimg[ri, ci] > gradimg[ri, ci-1] and \
                gradimg[ri, ci] > gradimg[ri, ci+1]:
                    edgeimg[ri, ci] = 1
    return edgeimg


def getBiMaxGraph(graph):
    gh, gw = graph.shape
    newgraph = np.zeros_like(graph)
    for ri in xrange(gw):
        row = graph[ri, :]
        maxcoli = np.argmax(row)
        col = graph[:, maxcoli]
        maxrowi = np.argmax(col)
        if maxrowi == ri and graph[ri, maxcoli] > 0:
            newgraph[ri, maxcoli] = graph[ri, maxcoli]
    
    return newgraph

def getLines(graph):
    imgh, imgw = graph.shape
    xx = []
    for ri in xrange(imgh):
        for ci in xrange(imgw):
            if graph[ri, ci] > 0:
                xx.append([ri, ci])
    xx = np.asarray(xx)
    return xx


def filterCrossLines(graph, xxtmp):
    xx = np.copy(xxtmp.T)
    lnum = xx.shape[1]
    for i1 in xrange(lnum-1):
        line1 = xx[:, i1]
        if line1[0]==0:
            continue
        v1 = graph[line1[0], line1[1]]
        for i2 in xrange(i1+1, lnum):
            line2 = xx[:, i2]
            if line2[0]==0:
                continue
            v2 = graph[line2[0], line2[1]]
            if ((line1[0]-line2[0])*(line1[1]-line2[1])<0):
                if v1>v2:
                    line2[0] = 0
                else:
                    line1[0] = 0
                    break
    xxnew = []
    for i1 in xrange(lnum):
        line = xx[:, i1]
        if line[0]==0:
            continue
        xxnew.append(line)
    xxnew = np.asarray(xxnew)
    
    return xxnew
    
fd = 'img/'
fns = ['1.bmp', '2.bmp', '3.bmp', '4.bmp', '5.bmp', '6.bmp', '7.bmp']
gimg = cv.imread(fd+fns[2], cv.CV_LOAD_IMAGE_GRAYSCALE)
gimg = cv.resize(gimg, (gimg.shape[1]*5/10, gimg.shape[0]))
imgh, imgw = gimg.shape
sblx = funcs.siSobelX_U8(gimg)
#sblx[-1::-1, :] = sblx[0:, :]
#sblx = get_edge(sblx)
segnum = 1
segimgh = imgh / segnum


wratio = [24, 27]
wratio = [0, 100] #35
#wratio = [0, 100]
for i in xrange(0, segnum):
    segimg = np.copy(sblx[segimgh*i:segimgh*(i+1), imgw*wratio[0]/100:imgw*wratio[1]/100])
    segimg2 = np.copy(gimg[segimgh*i:segimgh*(i+1), imgw*wratio[0]/100:imgw*wratio[1]/100])
#    cv.imshow('test', segimg)
    
    seggraph = generateGraph(segimg)
#    siPrintArray2D('%3d', seggraph)
    if 1:
        nmsgraph = doNMS(seggraph)
    elif 0:
        nmsgraph = seggraph
    else:
        nmsgraph = getBiMaxGraph(seggraph)
    
    xx1 = getLines(nmsgraph)
    xx1_1 = filterCrossLines(nmsgraph, xx1)
    siPrintArray2D('%4d', nmsgraph)
    xx2 = findLinesByDP(nmsgraph)
#    xx2 = filterRepeatLine(nmsgraph, xx2)
    siPrintArray2D('%3d', xx2)
    
    cimg1 = showBiMaxConnection(segimg2, segimg, nmsgraph, xx1)
    cimg2 = showBiMaxConnection(segimg2, segimg, nmsgraph, xx2)
    cimg1_1 = showBiMaxConnection(segimg2, segimg, nmsgraph, xx1_1)
    allimg = np.append(cimg1, cimg2[-1::-1, :], axis=0)
#    allimg = np.append(cimg1, cimg2, axis=0)
    allimg = np.append(allimg, cimg1_1, axis=0)
    cv.imshow('markline_max', allimg)
    cv.waitKey(0)



