# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 13:23:55 2014

@author: mzhang
"""
import numpy
import theano
import theano.tensor as T


def z(x, y):
    z = T.prod(x) * y
    return z

xx = T.imatrix('xx')
yy = T.ivector('yy')
wgts = T.fvector('wgts')

def set_value_at_position(x, y, onewgt):
    return z(x, y) * onewgt, x
#    return calcz(onex, oney) * onewgt

result, updates = theano.scan(fn=set_value_at_position,
                              outputs_info=None,
                              sequences=[xx, yy, wgts],
                              non_sequences=None)

r1, r2 = result
value_all = theano.function(inputs=[xx, yy, wgts], outputs=r1)


# test
xxmat = numpy.asarray([[1, 1], [2, 3]], dtype=numpy.int32)
yyvec = numpy.asarray([1, 2], dtype=numpy.int32)
wgts = numpy.asarray([0.5, 0.5], dtype=numpy.float32)
aa = value_all(xxmat, yyvec, wgts)
print aa
