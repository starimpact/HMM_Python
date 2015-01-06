# -*- coding: utf-8 -*-
"""
Created on Wed Nov 12 12:43:46 2014

@author: mzhang
"""

import cPickle
import numpy as np


sigma = cPickle.load(open('/Users/mzhang/work/HMM/sigma.bin', 'rb'))
h, w = sigma.shape
diff0 = sigma - sigma.T

u, s, v = np.linalg.svd(sigma)
s += 1e-1
#usv = np.dot(u, np.dot(np.diag(s), v))
usv = np.dot(u, np.dot(np.diag(s), u.T))
diff = usv - sigma

print np.sum(np.abs(diff0)), np.sum(np.abs(diff)), np.linalg.det(sigma), np.linalg.det(usv)

print u[0, :]
print v[:, 0]

