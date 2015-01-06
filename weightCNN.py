# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 10:10:08 2014

@author: mzhang

weighted cnn

"""


import cPickle
import gzip
import os
import sys
import time
import cv2

import numpy as np

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv
import lpfunctions as lpfuncs
import lpcr_func


#theano.config.exception_verbosity='high'
#theano.config.optimizer = 'None'

def showTensorData(tensorData):
    datafunc=theano.function(inputs=[],outputs=tensorData)
    data=datafunc()
#    print data
    return data


class HiddenLayer(object):
    def __init__(self, rng, input, n_in, n_out, W=None, b=None,
                 activation=T.tanh):
        """
        Typical hidden layer of a MLP: units are fully-connected and have
        sigmoidal activation function. Weight matrix W is of shape (n_in,n_out)
        and the bias vector b is of shape (n_out,).

        NOTE : The nonlinearity used here is tanh

        Hidden unit activation is given by: tanh(dot(input,W) + b)

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dmatrix
        :param input: a symbolic tensor of shape (n_examples, n_in)

        :type n_in: int
        :param n_in: dimensionality of input

        :type n_out: int
        :param n_out: number of hidden units

        :type activation: theano.Op or function
        :param activation: Non linearity to be applied in the hidden
                           layer
        """
        self.input = input

        # `W` is initialized with `W_values` which is uniformely sampled
        # from sqrt(-6./(n_in+n_hidden)) and sqrt(6./(n_in+n_hidden))
        # for tanh activation function
        # the output of uniform if converted using asarray to dtype
        # theano.config.floatX so that the code is runable on GPU
        # Note : optimal initialization of weights is dependent on the
        #        activation function used (among other things).
        #        For example, results presented in [Xavier10] suggest that you
        #        should use 4 times larger initial weights for sigmoid
        #        compared to tanh
        #        We have no info for other function, so we use the same as
        #        tanh.
        if W is None:
            W_values = np.asarray(rng.uniform(low=-np.sqrt(6. / (n_in + n_out)), high=np.sqrt(6. / (n_in + n_out)), size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4
            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)
        
        self.W = W
        self.b = b
        
        lin_output = T.dot(input, self.W) + self.b
        self.output = (lin_output if activation is None else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]



class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out, W = None, b = None):
        """ Initialize the parameters of the logistic regression

        :type input: theano.tensor.TensorType
        :param input: symbolic variable that describes the input of the
                      architecture (one minibatch)

        :type n_in: int
        :param n_in: number of input units, the dimension of the space in
                     which the datapoints lie

        :type n_out: int
        :param n_out: number of output units, the dimension of the space in
                      which the labels lie

        """

        # initialize with 0 the weights W as a matrix of shape (n_in, n_out)
        if W is None:
            W = theano.shared(value=np.zeros((n_in, n_out), dtype=theano.config.floatX), name='W', borrow=True)
        
        # initialize the baises b as a vector of n_out 0s
        if b is None:
            b = theano.shared(value=np.zeros((n_out,), dtype=theano.config.floatX), name='b', borrow=True)
        
        self.W = W
        self.b = b
        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
#        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        # parameters of the model
        self.params = [self.W, self.b]
    
    def negative_log_likelihood_test(self):
        return self.p_y_given_x
        
    def negative_log_likelihood(self, y):
        lt1 = T.log(self.p_y_given_x)
        lt2 = T.log(1.0 - self.p_y_given_x)
        ytmp = y.dimshuffle(0, 'x')
        return -T.mean(ytmp * lt1 + (1 - ytmp) * lt2)
        
    def negative_log_likelihood_weight(self, y, wgt):
        lt1 = T.log(self.p_y_given_x)
        lt2 = T.log(1.0 - self.p_y_given_x)
        ytmp = y.dimshuffle(0, 'x')
        wgttmp = wgt.dimshuffle(0, 'x')
        
#        return -T.mean(ytmp * lt1 + (1 - ytmp) * lt2)
        return -T.sum(wgttmp*(ytmp * lt1 + (1 - ytmp) * lt2))/T.sum(wgt)


class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2), W = None, b = None):
        """
        Allocate a LeNetConvPoolLayer with shared variable internal parameters.

        :type rng: numpy.random.RandomState
        :param rng: a random number generator used to initialize weights

        :type input: theano.tensor.dtensor4
        :param input: symbolic image tensor, of shape image_shape

        :type filter_shape: tuple or list of length 4
        :param filter_shape: (number of filters, num input feature maps,
                              filter height,filter width)

        :type image_shape: tuple or list of length 4
        :param image_shape: (batch size, num input feature maps,
                             image height, image width)

        :type poolsize: tuple or list of length 2
        :param poolsize: the downsampling (pooling) factor (#rows,#cols)
        """

        assert image_shape[1] == filter_shape[1]
        self.input = input

        # there are "num input feature maps * filter height * filter width"
        # inputs to each hidden unit
        fan_in = np.prod(filter_shape[1:])
        # each unit in the lower layer receives a gradient from:
        # "num output feature maps * filter height * filter width" /
        #   pooling size
        fan_out = (filter_shape[0] * np.prod(filter_shape[2:]) /
                   np.prod(poolsize))
        # initialize weights with random weights
        if W is None:
            W_bound = np.sqrt(6. / (fan_in + fan_out))
            W = theano.shared(np.asarray(rng.uniform(low=-W_bound, high=W_bound, size=filter_shape), dtype=theano.config.floatX), borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        if b is None:
            b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, borrow=True)
        
        self.W = W
        self.b = b
        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W, filter_shape=filter_shape, image_shape=image_shape)
        self.conv_out = conv_out
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out, ds=poolsize, ignore_border=True)
        
        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.sigmoid(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        # store parameters of this layer
        self.params = [self.W, self.b]


def buildWeightCNN(ishape, batch_size, nkerns=4, h_out=16):
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    rng = np.random.RandomState(23455)
    
#    nkerns = 8 #4 #kernel number
    filtersize = 5 #kernel filter size
    poolsize = 2 #pooling size
#    h_out = 128 #16 #hidden layer output number
    
    print 'nkerns:', nkerns, ', filtersize:', filtersize, ', poolsize:', poolsize, ', h_out:', h_out
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')
    wgt = T.fvector('wgt')
    
    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[1]))
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns, 1, filtersize, filtersize), poolsize=(poolsize, poolsize))

    layer1_input = layer0.output.flatten(2)
    l1_inputshape = ((ishape[0] - filtersize + 1) / poolsize, (ishape[1] - filtersize + 1) / poolsize)
    l1_input_mod = ((ishape[0] - filtersize + 1) % poolsize, (ishape[1] - filtersize + 1) % poolsize)
    print 'l1_inputshape:', l1_inputshape, l1_input_mod
    # construct a fully-connected sigmoidal layer
    layer1 = HiddenLayer(rng, input=layer1_input, n_in=nkerns * l1_inputshape[0] * l1_inputshape[1], 
                         n_out=h_out, activation=T.nnet.sigmoid)
    
    # classify the values of the fully-connected sigmoidal layer
    layer2 = LogisticRegression(input=layer1.output, n_in=h_out, n_out=1)
    
    # the cost we minimize during training is the NLL of the model
    cost_train_wegit = layer2.negative_log_likelihood_weight(y, wgt)
    cost_train = layer2.negative_log_likelihood(y)
    
    cost_test = layer2.negative_log_likelihood_test()
    
    params = layer2.params + layer1.params + layer0.params

    return cost_train, cost_train_wegit, params, x, y, wgt, cost_test
    

def buildWeightCNN2(ishape, batch_size):
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    rng = np.random.RandomState(23455)
    
    nkerns0 = 4 #kernel number
    nkerns1 = 8 #kernel number
    filtersize = 5 #kernel filter size
    poolsize = 2 #pooling size
    h_out = 16 #hidden layer output number
    
    print 'nkerns0:', nkerns0, 'nkerns1:', nkerns1, ', filtersize:', filtersize, ', poolsize:', poolsize, ', h_out:', h_out
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')
    wgt = T.fvector('wgt')
    
    # Reshape matrix of rasterized images of shape (batch_size,28*28)
    # to a 4D tensor, compatible with our LeNetConvPoolLayer
    layer0_input = x.reshape((batch_size, 1, ishape[0], ishape[1]))
    layer0 = LeNetConvPoolLayer(rng, input=layer0_input,
            image_shape=(batch_size, 1, ishape[0], ishape[1]),
            filter_shape=(nkerns0, 1, filtersize, filtersize), poolsize=(poolsize, poolsize))
    
    ishape1 = [(ishape[0]-filtersize+1)/poolsize, (ishape[1]-filtersize+1)/poolsize]
    print 'ishape1:', ishape1
    layer1 = LeNetConvPoolLayer(rng, input=layer0.output,
            image_shape=(batch_size, nkerns0, ishape1[0], ishape1[1]),
            filter_shape=(nkerns1, nkerns0, filtersize, filtersize), poolsize=(poolsize, poolsize))
    
    layer2_input = layer1.output.flatten(2)
    l2_inputshape = ((ishape1[0] - filtersize + 1) / poolsize, (ishape1[1] - filtersize + 1) / poolsize)
    l2_input_mod = ((ishape1[0] - filtersize + 1) % poolsize, (ishape1[1] - filtersize + 1) % poolsize)
    print 'l2_inputshape:', l2_inputshape, l2_input_mod
    # construct a fully-connected sigmoidal layer
    layer2 = HiddenLayer(rng, input=layer2_input, n_in=nkerns1 * l2_inputshape[0] * l2_inputshape[1], 
                         n_out=h_out, activation=T.nnet.sigmoid)
    
    # classify the values of the fully-connected sigmoidal layer
    layer3 = LogisticRegression(input=layer2.output, n_in=h_out, n_out=1)
    
    # the cost we minimize during training is the NLL of the model
    cost_train_wegit = layer3.negative_log_likelihood_weight(y, wgt)
    cost_train = layer3.negative_log_likelihood(y)
    
    cost_test = layer3.negative_log_likelihood_test()
    
    params = layer3.params + layer2.params + layer1.params + layer0.params

    return cost_train, cost_train_wegit, params, x, y, wgt, cost_test


#def getsamples_scales(lpinfo_list, imgBatchSize, batch_size, minibatch_index):
#    lpinfo_patch = lpinfo_list[minibatch_index * imgBatchSize:(minibatch_index + 1) * imgBatchSize]
#    datasets = lpcr_func.get_3sets_data_from_lpinfo_multiscale(lpinfo_patch, stdsize=ishape, sizeratio=(5., 0., 1.))
#    train_set_x, train_set_y = datasets[0]
#    valid_set_x, valid_set_y = datasets[1]
#    test_set_x, test_set_y = datasets[2]
##        print train_set_x.shape, train_set_y.shape
#    train_num, train_dim = train_set_x.shape
#    if train_set_x.shape[0] < batch_size:
#        addnum = batch_size - train_num
#        train_set_x = np.append(train_set_x, np.zeros((addnum, train_dim), dtype=train_set_x.dtype), axis=0)
#        train_set_y = np.append(train_set_y, np.zeros(addnum, dtype=train_set_y.dtype)-1, axis=0)
#        print 'add data: %d/%d->%d.....'%(batch_size, train_num, addnum)
##                continue
##        print train_set_x.shape, train_set_y.shape
#    tmptrainx = train_set_x[:batch_size, :]
#    tmptrainy = train_set_y[:batch_size]
#    posnum = np.sum(tmptrainy==1)
#    negnum = np.sum(tmptrainy==0)
#    allnum = posnum + negnum
##        print posnum, negnum, allnum
#    tmpwgt = np.zeros_like(tmptrainy, dtype=np.float32)
#    tmpwgt[tmptrainy==1] = 0.5 / posnum
#    tmpwgt[tmptrainy==0] = 0.5 / negnum
#    
#    return tmptrainx, tmptrainy, tmpwgt


def image_batch_training(lpinfo_list_tmp, ishape, batch_size, usewgt, train_model_weight, train_model, test_model, params, cnnparamsfile, sampletype=1):
    datasets = lpcr_func.get_3sets_data_from_lpinfo_multiscale(lpinfo_list_tmp, stdsize=ishape, sizeratio=(5., 0., 1.), sampletype=sampletype)
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]
    
    posnum = np.sum(train_set_y==1)
    negnum = np.sum(train_set_y==0)
    wgtall = np.zeros_like(train_set_y, dtype=np.float32)
    wgtall[train_set_y==1] = 0.5 / posnum
    wgtall[train_set_y==0] = 0.5 / negnum
    
    n_train_batches = train_set_x.shape[0] / batch_size
    n_test_batches = test_set_x.shape[0] / batch_size
    print 'n_train_batches:', n_train_batches, ', n_test_batches:', n_test_batches
    
    for batchidx in xrange(n_train_batches):
#            iter = (epoch - 1) * n_train_batches + batchidx
#            print tmpwgt
        tmptrainx = train_set_x[batchidx*batch_size:(batchidx+1)*batch_size, :]
        tmptrainy = train_set_y[batchidx*batch_size:(batchidx+1)*batch_size]
        if usewgt:
            tmpwgt1 = wgtall[batchidx*batch_size:(batchidx+1)*batch_size]
            cost = train_model_weight(tmptrainx, tmptrainy, tmpwgt1)
        else:
            cost = train_model(tmptrainx, tmptrainy)
        
    
#        test_cost = [test_model(test_set_x[tbi*batch_size:(tbi+1)*batch_size, :], test_set_y[tbi*batch_size:(tbi+1)*batch_size]) for tbi in xrange(n_test_batches)]
    rightnumall = [0, 0]
    numall = [0, 0]
    test_cost = 0
    for tbi in xrange(n_test_batches):
        tmp_test = test_set_x[tbi*batch_size:(tbi+1)*batch_size, :]
        tmp_y = test_set_y[tbi*batch_size:(tbi+1)*batch_size]
        ret = test_model(tmp_test)
        
        cost = np.sum(ret[tmp_y==0])
        cost += np.sum(1.0-ret[tmp_y==1])
        test_cost += cost
        
        posnum = np.sum(tmp_y==1)
        posret = ret[tmp_y==1]
        rightposnum = np.sum(posret>0.5)
        rightnumall[0] += rightposnum
        numall[0] += posnum
        
        negnum = np.sum(tmp_y==0)
        negret = ret[tmp_y==0]
        rightnegnum = np.sum(negret<0.5)
        rightnumall[1] += rightnegnum
        numall[1] += negnum
    
    cPickle.dump(params, open(cnnparamsfile, 'wb'))
    print 'cnn param is saved into:', cnnparamsfile
    
    return numall, rightnumall, test_cost
    

def train(lpinfo_list, batch_size=100, ishape=(32, 14), nkerns=4, h_out=16, sampletype=1, cnnparamsfile=None, cnnparamsfile_restore=None):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: int
    :param nkerns: number of kernels on each layer
    
    :type h_out: int
    :param h_out: number of units on hidden layer
    
    :type sampletype: int
    :param sampletype: sampling type
    """
    usewgt = True
    learning_rate = 0.1
    n_epochs = 100
    
    image_num = len(lpinfo_list)
    image_batch_size = 400
    image_batch_num = image_num / image_batch_size
    
    lpinfo_list_rnd = np.random.permutation(lpinfo_list)
    
    if usewgt:
        print 'training with weighted sample...'
    else:
        print 'training without weighted sample...'
        
    
    print 'learning_rate:',learning_rate,', max_epochs:',n_epochs,', batch_size:',batch_size
    print 'image_batch_size:', image_batch_size, ', image_batch_num:', image_batch_num
    
    
    cost_train, cost_train_weight, params, x, y, wgt, cost_test = buildWeightCNN(ishape, batch_size, nkerns, h_out)
#    cost_train, cost_train_weight, params, x, y, wgt, cost_test = buildWeightCNN2(ishape, batch_size)
    
    if cnnparamsfile_restore is not None:
        print 'set model from %s....'%(cnnparamsfile_restore)
        params_trained = cPickle.load(open(cnnparamsfile_restore, 'rb'))
        updates = []
        for param_i, trained_i in zip(params, params_trained):
            updates.append((param_i, trained_i))
        
        set_model = theano.function([], [], updates=updates)
        set_model()
    
    if usewgt:
        grads = T.grad(cost_train_weight, params)
    else:
        grads = T.grad(cost_train, params)
    
    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))
    
    train_model = None
    train_model_weight = None
    if usewgt:
        train_model_weight = theano.function(inputs=[x, y, wgt], outputs=cost_train_weight, updates=updates)
    else:
        train_model = theano.function(inputs=[x, y], outputs=cost_train, updates=updates)
    
    test_model = theano.function(inputs=[x], outputs=cost_test)
    
    
    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters

    start_time = time.clock()
    
    epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        numall = [0, 0]
        rightnumall = [0, 0]
        test_cost = 0
        for img_batchidx in xrange(image_batch_num):
            
            if img_batchidx == image_batch_num-1:
                lpinfo_list_tmp = lpinfo_list_rnd[img_batchidx*image_batch_size:]
            else:
                lpinfo_list_tmp = lpinfo_list_rnd[img_batchidx*image_batch_size:(img_batchidx+1)*image_batch_size]
            print 'loading batch image set data %d/%d, num:%d...'%(img_batchidx+1, image_batch_num, len(lpinfo_list_tmp))
            numallone, rightnumallone, test_costone = \
                                image_batch_training(lpinfo_list_tmp, ishape, batch_size, \
                                usewgt, train_model_weight, train_model, test_model, params, cnnparamsfile, sampletype)
            numall[0] += numallone[0]
            numall[1] += numallone[1]
            rightnumall[0] += rightnumallone[0]
            rightnumall[1] += rightnumallone[1]
            test_cost += test_costone
            print '++++ img_batch:%d/%d'%(img_batchidx+1, image_batch_num), \
                '+:%d/%d -:%d/%d'%(rightnumallone[0], numallone[0], rightnumallone[1], numallone[1]), \
                'test_cost:%.6f'%(test_costone/np.sum(numallone))
            
        print '---------- epoch:%d/%d'%(epoch, n_epochs), \
            '+:%d/%d -:%d/%d'%(rightnumall[0], numall[0], rightnumall[1], numall[1]), \
            'test_cost:%.6f'%(test_cost/np.sum(numall))
        print
#        print 'epoch:%d/%d'%(epoch, n_epochs), '  test_cost:', np.mean(test_cost)
    
    end_time = time.clock()
    print('Optimization complete.')
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))



def fillObsChain(lpinfo_list, stdsize, nkerns=4, h_out=16, sampletype=1, cnnparamsfile=None):
    print 'fill the obs_chain using scnn...'
    
    params_trained = cPickle.load(open(cnnparamsfile, 'rb'))
    cost_train, cost_train_weight, params, x, y, wgt, cost_test = buildWeightCNN(stdsize, 1, nkerns, h_out)
    
    print 'set model from %s....'%(cnnparamsfile)
    
    updates = []
    for param_i, trained_i in zip(params, params_trained):
        updates.append((param_i, trained_i))
    
    set_model = theano.function([], [], updates=updates)
    set_model()
    
    test_model = theano.function([x], cost_test)
    veclen = stdsize[0]*stdsize[1]
    halfw = stdsize[1]/2
    for lp in lpinfo_list:
        gimg = lp.charobj.grayimg
        mask = np.zeros_like(gimg)
        obs_chain = np.zeros(gimg.shape[1], dtype=np.float32)
        for wi in xrange(gimg.shape[1]-stdsize[1]):
            imgpart = gimg[:, wi:wi+stdsize[1]]
            imgrsz = cv2.resize(imgpart, (stdsize[1], stdsize[0])) #
            imgvec = np.reshape(imgrsz, veclen)
#            fimgvec = imgvec.astype(np.float32)
#            fimgvec = lpcr_func.normalize_data(fimgvec)
            fimgvec = lpcr_func.normalize_img_data_to_0_1_c(imgvec, 10)
            score = test_model(fimgvec.reshape(1, veclen))
            oriscore = score * 255
            obs_chain[wi+halfw] = oriscore
            mask[:, wi+halfw] = int(oriscore)
            
#            imgvec = fimgvec * 255
#            imgvec = imgvec.astype(np.uint8);
#            cv2.imshow('rsz', imgvec.reshape(stdsize))
#            cv2.waitKey(0)
        lp.charobj.obs_chain = obs_chain
        if 0:
            allimg = gimg / 2 + mask / 2
            cv2.imshow('result', allimg)
            cv2.waitKey(10)
        
#        break



def saveCNNParam2TXT(prefix, cnnparamsfile = 'wgtcnn.params.bin'):
    
    params_trained = cPickle.load(open(cnnparamsfile, 'rb'))
    cnnparamstxt = cnnparamsfile + '.txt'
    txtfile = open(cnnparamstxt, 'w')
    for tdi, tdata in enumerate(params_trained):
        data = showTensorData(tdata)
#        print tdata
#        print data.shape, data
        lshape = len(data.shape)
        if lshape == 1:
            txtfile.write('//dim:%d\n'%(data.shape[0]))
            txtfile.write('float gafParams%s%d_[%d] = '%(prefix, tdi, data.shape[0]))
            txtfile.write('{')
            for ri in xrange(data.shape[0]):
                odtxt = '%.6ff'%(data[ri])
                if ri < data.shape[0] - 1:
                    odtxt += ', '
                txtfile.write(odtxt)
                
            txtfile.write('};\n\n')
            
        if lshape == 2:
            data = data.T
            txtfile.write('//dim:%dx%d\n'%(data.shape[0], data.shape[1]))
            txtfile.write('float gafParams%s%d_[%d] = '%(prefix, tdi, data.shape[0] * data.shape[1]))
            txtfile.write('{')
            for ri in xrange(data.shape[0]):
#                txtfile.write('{')
                for ci in xrange(data.shape[1]):
                    odtxt = '%.6ff'%(data[ri, ci])                    
                    if ci < data.shape[1] - 1:
                        odtxt += ', '
                    txtfile.write(odtxt)
                if ri < data.shape[0] - 1:
#                    txtfile.write('}, ')
                    txtfile.write(', ')
#                else:
#                    txtfile.write('}')
            txtfile.write('};\n\n')
        
        if lshape == 4:
            txtfile.write('//dim:%dx%dx%dx%d\n'%(data.shape[0], data.shape[1], data.shape[2], data.shape[3]))
            txtfile.write('float gafParams%s%d_[%d] = '%(prefix, tdi, data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]))
            txtfile.write('{')            
            for d0 in xrange(data.shape[0]):
#                txtfile.write('{')
                for d1 in xrange(data.shape[1]):
#                    txtfile.write('{')
                    for d2 in xrange(data.shape[2]):
#                        txtfile.write('{')
                        for d3 in xrange(data.shape[3]):
                            odtxt = '%.6ff'%(data[d0, d1, d2, d3])
                            if d3 < data.shape[3] - 1:
                                odtxt += ', '
                            txtfile.write(odtxt)
                        if d2 < data.shape[2] - 1:
#                            txtfile.write('}, ')
                            txtfile.write(', ')
#                        else:
#                            txtfile.write('}')
                    if d1 < data.shape[1] - 1:
#                        txtfile.write('}, ')
                        txtfile.write(', ')
#                    else:
#                        txtfile.write('}')
                if d0 < data.shape[0] - 1:
#                    txtfile.write('}, ')
                    txtfile.write(', ')
#                else:
#                    txtfile.write('}')
            txtfile.write('};\n\n')

    txtfile.close()
    print 'cnn params is saved into', cnnparamstxt, '.'

if 0:
    stdshape = (28, 14)
    neednum = 1000
    folderpath = '/Volumes/ZMData1/LPR_TrainData/new/'
    lpinfo_list, width_hist = lpfuncs.getall_lps2(folderpath, neednum, stdshape[0], ifstrech=False)
    print len(lpinfo_list)
    train(lpinfo_list, batch_size=200, ishape=stdshape)



