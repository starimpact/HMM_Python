# -*- coding: utf-8 -*-
"""
Created on Wed Jan 14 10:54:49 2015

@author: mzhang
"""

import numpy as np

import theano
import theano.tensor as T
import cPickle


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None):
        self.input = input
        
        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)), dtype=theano.config.floatX)
            W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        
        self.W = W
        self.b = b
        
        lin_output = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
        self.output = lin_output
        
        # parameters of the model
        self.params = [self.W, self.b]



class SquareRegression(object):
    def __init__(self, input, n_in, n_out, W = None, b = None):

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        
        # initialize the baises b as a vector of n_out 0s
        if b is None:
            b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        
        self.W = W
        self.b = b
        
        self.y_given_x = T.dot(input, self.W) + self.b
        
        # parameters of the model
        self.params = [self.W, self.b]
        
        
    def square_lost(self, y):
        outy = self.y_given_x
        loss = (outy - y) * (outy - y)
        loss_sum = T.sum(loss, axis=1)
        loss_mean = T.mean(loss_sum)
        return loss_mean
        
    def test(self):
        return self.y_given_x
        

def create_RegNN(hiddenparam):
    x = T.matrix('x')
    y = T.matrix('y')
    
    rng = np.random.RandomState(23455)
    
    hidin, hidout = hiddenparam
    layer0 = HiddenLayer(rng, input=x, n_in=hidin, n_out=hidout)
    layer1 = SquareRegression(input=layer0.output, n_in=hidout, n_out=2)
    
    cost = layer1.square_lost(y)

    params = layer1.params + layer0.params

    return x, y, params, cost
    
def create_TestRegNN(hiddenparam):
    x = T.matrix('x')
    
    rng = np.random.RandomState(23455)
    
    hidin, hidout = hiddenparam
    layer0 = HiddenLayer(rng, input=x, n_in=hidin, n_out=hidout)
    layer1 = SquareRegression(input=layer0.output, n_in=hidout, n_out=2)
    
    outy = layer1.test()

    params = layer1.params + layer0.params

    return x, params, outy
    
def train(trainx, trainy, batchsize = 100):
    print 'training ...'
    regnnfn = 'regnn.bin'
    
    learn_rate = 0.1
    
    splnum, xdim = trainx.shape
    randidx = range(splnum)
    randidx = np.random.permutation(randidx)
    trainx = trainx[randidx, :]
    trainy = trainy[randidx, :]
    
    hiddenparam = [xdim, xdim*2]
    x, y, params, cost = create_RegNN(hiddenparam)
    
    grads = T.grad(cost, params)
    updatelist = []
    for oneparam, onegrad in zip(params, grads):
        updatelist.append((oneparam, oneparam-learn_rate*onegrad))
    
    trainfunc = theano.function(inputs=[x, y], outputs=cost, updates=updatelist)
    
    if 1:
        print 'set parameters using file', regnnfn
        setlist = []
        oldparams = cPickle.load(open(regnnfn))
        for oneparam, oldone in zip(params, oldparams):
            setlist.append((oneparam, oldone))
        setfunc = theano.function(inputs=[], updates=setlist)
        setfunc()
    
        
    maxloop = 100
    batchnum = splnum / batchsize
    
    costall = np.zeros(batchnum, dtype=np.float32)
    for li in xrange(maxloop):
        for pi in xrange(batchnum):
            batchx = trainx[pi*batchsize:(pi+1)*batchsize, :]
            batchy = trainy[pi*batchsize:(pi+1)*batchsize, :]
            
            costval = trainfunc(batchx, batchy)
            costall[pi] = costval
        
        costmean = np.mean(costall)
        cPickle.dump(params, open(regnnfn, 'wb'))
        infotimp = 'params is saved into ' + regnnfn
        print li, ':', costmean, infotimp


def test(trainx):
    regnnfn = 'regnn.bin'
    print 'set parameters using file', regnnfn
    
    splnum, xdim = trainx.shape
    hiddenparam = [xdim, xdim*2]
    x, params, outy = create_TestRegNN(hiddenparam)
    
    yfunc = theano.function([x], outy)
    
    setlist = []
    oldparams = cPickle.load(open(regnnfn))
    for oneparam, oldone in zip(params, oldparams):
        setlist.append((oneparam, oldone))
    setfunc = theano.function(inputs=[], updates=setlist)
    setfunc()

    yval = yfunc(trainx)

    return yval
