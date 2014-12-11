# -*- coding: utf-8 -*-
"""
Created on Thu Nov 20 11:18:57 2014

@author: mzhang
"""
import time 
import numpy
import pp

valist = (200000, 300000, 400000, 500000)

def logeval(n2):
    sumall = 0
    for i in xrange(1, n2):
        sumall += numpy.log(i) + numpy.log(i)
    return sumall

tms = time.time()
for val in valist:
    result1 = logeval(val)
tme = time.time()
print 'origin time is:', tme - tms

job_server = pp.Server(ncpus=2)
print "Starting pp with", job_server.get_ncpus(), "workers"


joblist = []
for val in valist:
    job1 = job_server.submit(logeval, (val,), (), ('numpy', ))
    joblist.append(job1)


tms = time.time()
for job in joblist:
    result = job()
tme = time.time()
print 'pp time is:', tme - tms

job_server.print_stats()

