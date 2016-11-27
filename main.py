#!/usr/bin/env python
# encoding: utf-8

import mxnet as mx
import sys
import numpy as np
import time
from collections import namedtuple
import logging

def readData(lldFeatscpFile, lldLabelscpFile, nrows):
    emotion = {'A':0, 'E':1, 'N':2, 'P':3, 'R':4}   # Emotion Dictionary

    lldFeatscp = open(lldFeatscpFile, 'r')
    lldLabelscp = open(lldLabelscpFile, 'r')
    lldX = np.zeros((nrows, 384))
    lldY = np.zeros(nrows)

    uttIdx = 0
    csvfilename = lldFeatscp.readline().strip()
    labelfilename = lldLabelscp.readline().strip()
    labelfile = open(labelfilename, 'r')
    labelLine = labelfile.readline().strip()
    while (csvfilename != ''):
        csvfile = open(csvfilename, 'r')
        dimIdx = 0
        for line in csvfile.readlines():
            lldX[uttIdx][dimIdx] = float(line)
            dimIdx += 1

        if (labelLine == ''):
            print "Label Idx %d out of range in label file" % (uttIdx)
            exit()
        lldY[uttIdx] = emotion[labelLine.split()[1]]

        uttIdx += 1
        csvfile.close()
        csvfilename = lldFeatscp.readline().strip()
        labelLine = labelfile.readline().strip()

    if (labelLine != ""):
        print "Utterance Idx %d out of range in scp file" % (uttIdx)
        exit()

    print "Data %d utts" % uttIdx
    labelfile.close()
    lldFeatscp.close()
    lldLabelscp.close()

    return lldX, lldY

def DNN_def(nLayers, nodesPerLayer):
    Param = namedtuple('Param', ["weight", "bias"])

    data = mx.sym.Variable('data')
    label = mx.sym.Variable('softmax_label')

    hidden = data
    param = []
    for i in xrange(nLayers):
        param.append(Param(weight = mx.sym.Variable("l%d_weight" % (i+1)),
                           bias   = mx.sym.Variable("l%d_bias" % (i+1))
                          ))
        #next_layer_z = mx.sym.FullyConnected(data=hidden, num_hidden=nodesPerLayer,
        #                                     weight=param[i].weight, bias=param[i].bias,
        #                                     name="Linear%d" % (i+1))
        next_layer_z = mx.sym.FullyConnected(data=hidden, num_hidden=nodesPerLayer,
                                             name="Linear%d" % (i+1))
        hidden = mx.sym.Activation(next_layer_z, act_type="sigmoid", name="Activation%d"%(i+1))

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

    lldTrainX, lldTrainY = readData('./lld_train_feat.scp', './lld_train_label.scp', 9959)
    lldTestX, lldTestY   = readData('./lld_test_feat.scp', './lld_test_label.scp', 8257)

    nLayers = 3
    nodesPerLayer = 1024
    batchsize = 256

    train_iter = mx.io.NDArrayIter(lldTrainX, lldTrainY, batch_size = batchsize)
    test_iter = mx.io.NDArrayIter(lldTestX, lldTestY, batch_size = batchsize)

    sym = DNN_def(nLayers, nodesPerLayer)
    #module = mx.mod.Module(sym)
    #do_training(module, data_train, data_val)
    model = mx.model.FeedForward(
        ctx = mx.cpu(0),
        symbol = sym,
        num_epoch = 10,
        learning_rate = 0.1,
        momentum = 0.9,
        wd = 0.00001,
    )
    model.fit(
        X = train_iter,
        eval_data = test_iter,
        batch_end_callback = mx.callback.Speedometer(batchsize, 200),
    )

    print model.score(test_iter)*100, "%"
