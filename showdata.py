# -*- coding: utf-8 -*-
"""
Created on Wed Jun 25 10:42:55 2014

@author: mzhang
"""
import theano

def showTensorData(tensorData):
    datafunc=theano.function(inputs=[],outputs=tensorData)
    data=datafunc()
#    print data.shape,'\n',data
    print data
    return data
#    results,updates=theano.scan(do_show,non_sequences=[tensorData],n_steps=1)
#    print results