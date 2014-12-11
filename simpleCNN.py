# -*- coding: utf-8 -*-
"""
Created on Fri Nov 21 13:57:51 2014

@author: mzhang

This is good.
800 training samples is trained, and we get 2 erros over 200 test samples.
And, the 2 error license plates have weird number 1.

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
            W_values = np.asarray(rng.uniform(
                    low=-np.sqrt(6. / (n_in + n_out)),
                    high=np.sqrt(6. / (n_in + n_out)),
                    size=(n_in, n_out)), dtype=theano.config.floatX)
            if activation == theano.tensor.nnet.sigmoid:
                W_values *= 4

            W = theano.shared(value=W_values, name='W', borrow=True)

        if b is None:
            b_values = np.zeros((n_out,), dtype=theano.config.floatX)
            b = theano.shared(value=b_values, name='b', borrow=True)

        self.W = W
        self.b = b

        lin_output = T.dot(input, self.W) + self.b
        self.lin_output = lin_output
        self.output = (lin_output if activation is None
                       else activation(lin_output))
        # parameters of the model
        self.params = [self.W, self.b]



class LogisticRegression(object):
    """Multi-class Logistic Regression Class

    The logistic regression is fully described by a weight matrix :math:`W`
    and bias vector :math:`b`. Classification is done by projecting data
    points onto a set of hyperplanes, the distance to which is used to
    determine a class membership probability.
    """

    def __init__(self, input, n_in, n_out):
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
        self.W = theano.shared(value=np.zeros((n_in, n_out),
                                                 dtype=theano.config.floatX),
                                name='W', borrow=True)
        # initialize the baises b as a vector of n_out 0s
        self.b = theano.shared(value=np.zeros((n_out,),
                                                 dtype=theano.config.floatX),
                               name='b', borrow=True)

        # compute vector of class-membership probabilities in symbolic form
        self.p_y_given_x = T.nnet.sigmoid(T.dot(input, self.W) + self.b)
#        self.p_y_given_x = T.nnet.softmax(T.dot(input, self.W) + self.b)
        
        # parameters of the model
        self.params = [self.W, self.b]

    def negative_log_likelihood(self, y):
        lt1 = T.log(self.p_y_given_x)
        lt2 = T.log(1.0 - self.p_y_given_x)
        ytmp = y.dimshuffle(0, 'x')
        return -T.mean(ytmp * lt1 + (1 - ytmp) * lt2)
#        return -T.mean(T.log(self.p_y_given_x)[T.arange(y.shape[0]), y])
#        return -T.mean(logistic)



class LeNetConvPoolLayer(object):
    """Pool Layer of a convolutional network """

    def __init__(self, rng, input, filter_shape, image_shape, poolsize=(2, 2)):
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
        W_bound = np.sqrt(6. / (fan_in + fan_out))
        self.W = theano.shared(np.asarray(
            rng.uniform(low=-W_bound, high=W_bound, size=filter_shape),
            dtype=theano.config.floatX),
                               borrow=True)

        # the bias is a 1D tensor -- one bias per output feature map
        b_values = np.zeros((filter_shape[0],), dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values, borrow=True)

        # convolve input feature maps with filters
        conv_out = conv.conv2d(input=input, filters=self.W,
                filter_shape=filter_shape, image_shape=image_shape)
        self.conv_out = conv_out
        # downsample each feature map individually, using maxpooling
        pooled_out = downsample.max_pool_2d(input=conv_out,
                                            ds=poolsize, ignore_border=True)

        # add the bias term. Since the bias is a vector (1D array), we first
        # reshape it to a tensor of shape (1,n_filters,1,1). Each bias will
        # thus be broadcasted across mini-batches and feature map
        # width & height
        self.output = T.nnet.sigmoid(pooled_out + self.b.dimshuffle('x', 0, 'x', 'x'))
        
        # store parameters of this layer
        self.params = [self.W, self.b]


def buildCNN(batch_size, ishape):
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    rng = np.random.RandomState(23455)
    
    nkerns = 4 #kernel number
    filtersize = 5 #kernel filter size
    poolsize = 2 #pooling size
    h_out = 32 #hidden layer output number
    
    # allocate symbolic variables for the data
    
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')
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
    layer1 = HiddenLayer(rng, input=layer1_input, n_in=nkerns * l1_inputshape[0] * l1_inputshape[1], n_out=h_out, activation=T.nnet.sigmoid)
    
    # classify the values of the fully-connected sigmoidal layer
    layer2 = LogisticRegression(input=layer1.output, n_in=h_out, n_out=1)
    
    # the cost we minimize during training is the NLL of the model
    cost = layer2.negative_log_likelihood(y)
    
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer2.params + layer1.params + layer0.params
    
    return cost, params, x, y


    
def train(lpinfo_list, batch_size, ishape = (32, 14)):
    """ Demonstrates lenet on MNIST dataset

    :type learning_rate: float
    :param learning_rate: learning rate used (factor for the stochastic
                          gradient)

    :type n_epochs: int
    :param n_epochs: maximal number of epochs to run the optimizer

    :type dataset: string
    :param dataset: path to the dataset used for training /testing (MNIST here)

    :type nkerns: list of ints
    :param nkerns: number of kernels on each layer
    """
    learning_rate = 0.1
    n_epochs = 200
    showperiod = 800
    
    datasets = lpcr_func.get_3sets_data_from_lpinfo(lpinfo_list, stdsize=ishape, sizeratio=(4., 0., 1.))
    
    train_set_x, train_set_y = datasets[0]
    valid_set_x, valid_set_y = datasets[1]
    test_set_x, test_set_y = datasets[2]

    # compute number of minibatches for training, validation and testing
    n_train_batches = train_set_x.get_value(borrow=True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow=True).shape[0]
    n_test_batches = test_set_x.get_value(borrow=True).shape[0]
    n_train_batches /= batch_size
    n_valid_batches /= batch_size
    n_test_batches /= batch_size
    
    print 'trainset_size:',n_train_batches, ', validset_size:', n_valid_batches, ', testset_size:', n_test_batches
    print 'learning_rate:',learning_rate,', max_epochs:',n_epochs,', batch_size:',batch_size

    index = T.lscalar()  # index to a [mini]batch
    
    cost, params, x, y = buildCNN(batch_size, ishape)
    
    # create a list of gradients for all model parameters
    grads = T.grad(cost, params)
    
    # create a function to compute the mistakes that are made by the model
    test_model = theano.function([index], cost,
             givens={
                x: test_set_x[index * batch_size: (index + 1) * batch_size],
                y: test_set_y[index * batch_size: (index + 1) * batch_size]})


    updates = []
    for param_i, grad_i in zip(params, grads):
        updates.append((param_i, param_i - learning_rate * grad_i))

    train_model = theano.function([index], cost, updates=updates,
          givens={
            x: train_set_x[index * batch_size: (index + 1) * batch_size],
            y: train_set_y[index * batch_size: (index + 1) * batch_size]})

    ###############
    # TRAIN MODEL #
    ###############
    print '... training'
    # early-stopping parameters

    start_time = time.clock()

    epoch = 0
    while epoch < n_epochs:
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):

            iter = (epoch - 1) * n_train_batches + minibatch_index
            
            train_cost = train_model(minibatch_index)
#            print train_cost
            if iter % showperiod == 0:
                test_cost = [test_model(i) for i in xrange(n_test_batches)]
                print 'epoch:%d/%d'%(epoch, n_epochs), 'training @ iter = ', iter, '  test_cost:', np.mean(test_cost)
                cnnparamsfile = 'cnn.params.bin'
                cPickle.dump(params, open(cnnparamsfile, 'wb'))
                print 'cnn param is saved into:', cnnparamsfile
                
    end_time = time.clock()
    print('Optimization complete.')
    print >> sys.stderr, ('The code for file ' +
                          os.path.split(__file__)[1] +
                          ' ran for %.2fm' % ((end_time - start_time) / 60.))


def fillObsChain(lpinfo_list, stdsize):
    print 'fill the obs_chain using scnn...'
    
    cnnparamsfile = 'cnn.params.bin'
    params_trained = cPickle.load(open(cnnparamsfile, 'rb'))
    cost, params, x, y = buildCNN(1, stdsize)
    
    updates = []
    for param_i, trained_i in zip(params, params_trained):
        updates.append((param_i, trained_i))
    
    set_model = theano.function([], [], updates=updates)
    set_model()
    
    test_model = theano.function([x, y], cost)
    veclen = stdsize[0]*stdsize[1]
    yvalue = np.ones(1, dtype=np.int32)
    halfw = stdsize[1]/2
    for lp in lpinfo_list:
        gimg = lp.charobj.grayimg
        mask = np.zeros_like(gimg)
        obs_chain = np.zeros(gimg.shape[1], dtype=np.float32)
        for wi in xrange(gimg.shape[1]-stdsize[1]):
            imgpart = gimg[:, wi:wi+stdsize[1]]
            imgrsz = cv2.resize(imgpart, (stdsize[1], stdsize[0])) #resize image to 28x28
            imgvec = np.reshape(imgrsz, veclen)
            fimgvec = imgvec.astype(np.float32)
            fimgvec = lpcr_func.normalize_data(fimgvec)
            score = test_model(fimgvec.reshape(1, veclen), yvalue)
            oriscore = np.exp(-score) * 255
            obs_chain[wi+halfw] = oriscore
            mask[:, wi+halfw] = int(oriscore)
            
#            imgvec = fimgvec * 255
#            imgvec = imgvec.astype(np.uint8);
#            cv2.imshow('rsz', imgvec.reshape(stdsize))
#            cv2.waitKey(0)
        lp.charobj.obs_chain = obs_chain
        allimg = gimg / 2 + mask / 2
        cv2.imshow('result', allimg)
        cv2.waitKey(40)
        
#        break



def buildCNNtmp(batch_size, ishape):
    ######################
    # BUILD ACTUAL MODEL #
    ######################
    print '... building the model'
    
    rng = np.random.RandomState(23455)
    
    nkerns = 4 #kernel number
    filtersize = 5 #kernel filter size
    poolsize = 2 #pooling size
    h_out = 32 #hidden layer output number
    
    # allocate symbolic variables for the data
    
    x = T.matrix('x')   # the data is presented as rasterized images
    y = T.ivector('y')
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
    layer1 = HiddenLayer(rng, input=layer1_input, n_in=nkerns * l1_inputshape[0] * l1_inputshape[1], n_out=h_out, activation=T.nnet.sigmoid)
    
    # classify the values of the fully-connected sigmoidal layer
    layer2 = LogisticRegression(input=layer1.output, n_in=h_out, n_out=1)
    
    # the cost we minimize during training is the NLL of the model
    cost = layer2.negative_log_likelihood(y)
    
    
    # create a list of all model parameters to be fit by gradient descent
    params = layer2.params + layer1.params + layer0.params
    
    return cost, params, x, y, layer1_input, layer1.lin_output, layer1.output, layer2.p_y_given_x
    
def tmptest2(lp, stdsize):
    print 'fill the obs_chain using scnn...'
    
    cnnparamsfile = 'cnn.params.bin'
    params_trained = cPickle.load(open(cnnparamsfile, 'rb'))
    cost, params, x, y, layer1_input, layer1_lin_output, layer1_output, layer2_p_y_given_x = buildCNNtmp(1, stdsize)
    
    updates = []
    for param_i, trained_i in zip(params, params_trained):
        updates.append((param_i, trained_i))
    
    set_model = theano.function([], [], updates=updates)
    set_model()
    
    do_cov = theano.function([x], layer1_input)
    do_hid1pre = theano.function([x], layer1_lin_output)
    do_hid1 = theano.function([x], layer1_output)
    do_hid2 = theano.function([x], layer2_p_y_given_x)
    
    test_model = theano.function([x, y], cost)
    veclen = stdsize[0]*stdsize[1]
    yvalue = np.ones(1, dtype=np.int32)
    halfw = stdsize[1]/2
    
    gimg = lp.charobj.grayimg
    mask = np.zeros_like(gimg)
    obs_chain = np.zeros(gimg.shape[1], dtype=np.float32)
    for wi in xrange(gimg.shape[1]-stdsize[1]):
        imgpart = gimg[:, wi:wi+stdsize[1]]
        imgrsz = cv2.resize(imgpart, (stdsize[1], stdsize[0])) #resize image
        imgvec = np.reshape(imgrsz, veclen)
#        print imgpart
        fimgvec = imgvec.astype(np.float32)
#        print fimgvec
        fimgvec = lpcr_func.normalize_data(fimgvec)
#        print fimgvec
        imgvector = fimgvec.reshape(1, veclen)
        
        cov_result = do_cov(imgvector)
        hid1pre_result = do_hid1pre(imgvector)
        hid1_result = do_hid1(imgvector)
        hid2_result = do_hid2(imgvector)
        print cov_result[0, :]
        print hid1pre_result
        print 'hid1_result:', hid1_result
        print 'hid2_result:', hid2_result
        print 'log:', np.log(hid2_result)
        
        score = test_model(imgvector, yvalue)        
        print score
        
        oriscore = np.exp(-score) * 255
        print 'oriscore:', oriscore
        obs_chain[wi+halfw] = oriscore
        mask[:, wi+halfw] = int(oriscore)
        if wi >= 0:
            break
#            imgvec = fimgvec * 255
#            imgvec = imgvec.astype(np.uint8);
#            cv2.imshow('rsz', imgvec.reshape(stdsize))
#            cv2.waitKey(0)
    lp.charobj.obs_chain = obs_chain
    allimg = gimg / 2 + mask / 2
    cv2.imshow('result', allimg)
    cv2.waitKey(40)
    
def tmptest(lp, stdsize):
    print 'fill the obs_chain using scnn...'
    
    cnnparamsfile = 'cnn.params.bin'
    params_trained = cPickle.load(open(cnnparamsfile, 'rb'))
    cost, params, x, y, cnnRet_tmp = buildCNNtmp(1, stdsize)
    
#    print len(params_trained)
    ttt = showTensorData(params_trained[4])
    bbb = params_trained[5]
    bbb1 = bbb[:1]
    ttt1 = ttt[:1, :1, :, :]
    print ttt.shape, ttt1.shape
    print ttt1[0, 0, :, :]
    imginput = T.tensor4('imginput')
    conv_out = conv.conv2d(input=imginput, filters=ttt1)
    do_Conv = theano.function([imginput], conv_out)
    pooled_out = downsample.max_pool_2d(input=conv_out, ds=(2, 2), ignore_border=True)
    do_pool = theano.function([imginput], pooled_out)
    add_out = pooled_out + bbb1.dimshuffle('x', 0, 'x', 'x')
    do_add = theano.function([imginput], add_out)
    sigmoid_out = T.nnet.sigmoid(add_out)
    do_sigmoid = theano.function([imginput], sigmoid_out)
    
    updates = []
    for param_i, trained_i in zip(params, params_trained):
        updates.append((param_i, trained_i))
    
    set_model = theano.function([], [], updates=updates)
    set_model()
    
    test_model = theano.function([x, y], cost)
    veclen = stdsize[0]*stdsize[1]
    yvalue = np.ones(1, dtype=np.int32)
    halfw = stdsize[1]/2
    
    gimg = lp.charobj.grayimg
    mask = np.zeros_like(gimg)
    obs_chain = np.zeros(gimg.shape[1], dtype=np.float32)
    for wi in xrange(gimg.shape[1]-stdsize[1]):
        imgpart = gimg[:, wi:wi+stdsize[1]]
        imgrsz = cv2.resize(imgpart, (stdsize[1], stdsize[0])) #resize image
        imgvec = np.reshape(imgrsz, veclen)
#        print imgpart
        fimgvec = imgvec.astype(np.float32)
#        print fimgvec
        fimgvec = lpcr_func.normalize_data(fimgvec)
#        print fimgvec
        imgvector = fimgvec.reshape(1, veclen)
        
        score = test_model(imgvector, yvalue)
        
        imgvector2 = imgvector.reshape((1, 1, stdsize[0], stdsize[1]))
        
        imgvector2tmp = imgvector2[:1, :1, :5, :5]
        cov_result = do_Conv(imgvector2)
        pool_result = do_pool(imgvector2)
        sigmoid_result = do_sigmoid(imgvector2)
        add_result = do_add(imgvector2)
        print imgvector2tmp
        tmp111 = (ttt1[0, 0, -1::-1, -1::-1] * imgvector2tmp[0, 0, :, :])
        print tmp111
        print np.sum(tmp111)
        print cov_result.shape, pool_result.shape
        print cov_result[0, 0, :, :]
        print pool_result[0, 0, :, :]
        print add_result[0, 0, :, :]
        print sigmoid_result[0, 0, :, :]
        
        
        oriscore = np.exp(-score) * 255
        print 'oriscore:', oriscore
        obs_chain[wi+halfw] = oriscore
        mask[:, wi+halfw] = int(oriscore)
        if wi >= 0:
            break
#            imgvec = fimgvec * 255
#            imgvec = imgvec.astype(np.uint8);
#            cv2.imshow('rsz', imgvec.reshape(stdsize))
#            cv2.waitKey(0)
    lp.charobj.obs_chain = obs_chain
    allimg = gimg / 2 + mask / 2
    cv2.imshow('result', allimg)
    cv2.waitKey(40)

def saveCNNParam2TXT():
    cnnparamsfile = 'cnn.params.bin'
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
            txtfile.write('float gafParams%d_[%d] = '%(tdi, data.shape[0]))
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
            txtfile.write('float gafParams%d_[%d] = '%(tdi, data.shape[0] * data.shape[1]))
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
            txtfile.write('float gafParams%d_[%d] = '%(tdi, data.shape[0] * data.shape[1] * data.shape[2] * data.shape[3]))
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


#saveCNNParam2TXT()
#ishape = (32, 10) #cnn input shape
#lpimg_neednum = 100
#folderpath = '/Users/mzhang/work/LP Data2/'
#lpinfo_list = lpfuncs.getall_lps(folderpath, lpimg_neednum, ifstrech=False)
##train(lpinfo_list, batch_size=100, ishape=ishape)
#test(lpinfo_list, ishape)







