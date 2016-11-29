#!/usr/bin/env python
# encoding: utf-8

import mxnet as mx
import sys
import numpy as np
import time
from collections import namedtuple
import logging
from common import Data

def DNN_def(nLayers, nodesPerLayer):
    Param = namedtuple('Param', ["weight", "bias"]) # Currently, param(eters) are not used.
    param = []

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')

    hidden = data
    for i in xrange(nLayers):
        #param.append(Param(weight = mx.sym.Variable("l%d_weight" % (i+1)),
        #                   bias   = mx.sym.Variable("l%d_bias" % (i+1))
        #                  ))
        #next_layer_z = mx.sym.FullyConnected(data=hidden, num_hidden=nodesPerLayer,
        #                                     weight=param[i].weight, bias=param[i].bias,
        #                                     name="Linear%d" % (i+1))
        next_layer_z = mx.sym.FullyConnected(data=hidden, num_hidden=nodesPerLayer,
                                             name="Linear%d" % (i+1))
        hidden = mx.sym.Activation(next_layer_z, act_type="relu", name="Activation%d"%(i+1))
    hidden = mx.sym.FullyConnected(data = hidden ,num_hidden=5, name="output")
    sm = mx.sym.SoftmaxOutput(data=hidden, label=label, ignore_label=0, use_ignore=False,
                              name='softmax')
    return sm

#def get_initializer():
#    return mx.initializer.Xavier(magnitude=0.05)
#
#def do_traininit(module, data_train, data_val):
#
#    module.bind(data_shapes = data_train.provide_data,
#                label_shapes = data_train.provide_label,
#                for_training = True)
#    module.init_params(initializer=get_initializer())
#
#    module.init_optimizer(kvstore="local",
#                          optimizer="sgd",
#                          optimizer_params={'learning_rate'=0.01})
#
#    while True:
#        for data_batch in data_train:
#            module.forward_backward(data_batch)
#            module.update()


if __name__=="__main__":
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)

    # load data
    lld = Data('lld')
    lld.load_training_data()
    lld.load_test_data()

    # Define Network Architectures
    nLayers = 2
    nodesPerLayer = 1024
    batchsize = 256
    sym = DNN_def(nLayers, nodesPerLayer)

    #train_iter = mx.io.NDArrayIter(lld.feature_train, lld.label_train[:, 0], batch_size = batchsize)
    #test_iter  = mx.io.NDArrayIter(lld.feature_test, lld.label_test[:, 0], batch_size = batchsize)
    train_iter = mx.io.NDArrayIter(lld.feature_train, batch_size = batchsize)

    test_iter  = mx.io.NDArrayIter(lld.feature_test, batch_size = batchsize)
    train_iter.label = mx.io._init_data(lld.label_train[:, 0], allow_empty=True, default_name='softmax_label')
    test_iter.label  = mx.io._init_data(lld.label_test[:, 0], allow_empty=True, default_name='softmax_label')
    # train_iter.next()
    # print train_iter.getdata()[0].asnumpy()[:10,0]
    

    # Training the Model
    '''
    # To do
    #module = mx.mod.Module(sym)
    #do_training(module, data_train, data_val)
    '''
    model = mx.model.FeedForward(
        ctx = mx.cpu(0),
        symbol = sym,
        num_epoch = 10,
        learning_rate = 0.001,
        momentum = 0.9,
        wd = 0.00001,
    )
    model.fit(
        X = train_iter,
        eval_data = train_iter,
        eval_metric        = ['accuracy'],
        batch_end_callback = mx.callback.Speedometer(batchsize, 200),
    )

    # print model.predict(train_iter)
    print model.score(test_iter)*100, "%"
