import os, copy, datetime
import numpy as np
import chainer, datetime, os, json
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as C
from datahelper import GraphMes, DataHelper, dataset
from chainer import initializers
from chainer.training import extensions
from mylib.texthelper.format import pshape, pdata, ptree
from chainer.initializers.normal import Normal
from chainer import reporter, Variable, Chain, Parameter, link
from chainer import datasets, iterators, optimizers, training, serializers
from chainer.initializers.uniform import Uniform
from modules import RVec, NVec, TransE

def Settings():
    with open('../../model/settings.json') as f:
        settings = json.load(f)
    return settings

def train(root):
    settings = Settings()
    print("batch_size:"         , settings['batch_size'])
    print("epoch: "             , settings['epoch'])
    print("vecDims: "           , settings['vecDims'])
    print("learning_rate: "     , settings['learning_rate'])

    # train = "../../data/raw_small/train_id.txt"
    # valid = "../../data/raw_small/valid_id.txt"
    # info = "../../data/raw_small/info"
    
    train = "../../data/raw/train_id.txt"
    valid = "../../data/raw/valid_id.txt"
    info = "../../data/raw/info"
    
    trainSet, validSet, nodeCnt = dataset(train, valid, info)
    
    train_iterator  = chainer.iterators.SerialIterator(trainSet, settings['batch_size'], repeat=True)
    valid_iterator  = chainer.iterators.SerialIterator(validSet, settings['batch_size'], repeat=False)

    model = TransE(nodeCnt, settings)
    
    optimizer = optimizers.Adam(alpha=settings['alpha'],beta1=settings['beta1'],beta2=settings['beta1'],eps=settings['eps'])
    # optimizer = optimizers.SGD(settings['learning_rate'])
    optimizer.setup(model)
    updater = training.updater.StandardUpdater(train_iterator,optimizer)
    
    dt_now = datetime.datetime.now()
    time_stample = './results'+str(dt_now.day)+'_'+str(dt_now.hour)+'_'+str(dt_now.minute)+'_'+str(dt_now.second)+'/'
    trainer = training.Trainer(updater,(settings['epoch'], 'epoch'), out=time_stample)

    trainer.extend(extensions.Evaluator(valid_iterator, model))
    trainer.extend(extensions.LogReport())
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],'iteration',
                trigger=(1, 'iteration'), file_name='loss.png'))
    
    trainer.run()

if __name__ == "__main__":
    root = "../../data/"
    # files = ["../data/WNM_O50T50"]
    train(root)