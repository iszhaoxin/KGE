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


class RVec(link.Link):
    def __init__(self, counts, vecDims):
        super(RVec, self).__init__()
        with self.init_scope():
            initializer     = Uniform(6/np.sqrt(vecDims))
            self.vecDims    = vecDims
            self.edge2vec   = Parameter(initializer)
            self.edge2vec.initialize((counts, vecDims))
            self.edge2vec2   = F.normalize(self.edge2vec)
    def forward(self, indexs):
        vecs = F.embed_id(indexs, self.edge2vec2).reshape(-1, self.vecDims)
        return vecs
        
class NVec(link.Link):
    def __init__(self, counts, vecDims):
        super(NVec, self).__init__()
        with self.init_scope():
            initializer     = Uniform(6/np.sqrt(vecDims))
            self.counts     = counts
            self.vecDims    = vecDims
            self.node2vec   = Parameter(initializer)
            self._initialize_params()
    def _initialize_params(self):
        self.node2vec.initialize((self.counts, self.vecDims))
    def forward(self, indexs):
        self.nodeVecs = F.normalize(self.node2vec)
        vecs = F.embed_id(indexs, self.node2vec).reshape(-1, self.vecDims)    
        return vecs

class TransE(chainer.link.Chain):
    def __init__(self, nodeCnt, settings):
        super(TransE, self).__init__()
        with self.init_scope():
            self.settings   = settings
            self.margin     = settings['margin']
            self.nodeCnt    = nodeCnt
            self.random     = np.random.randint
            self.nVec       = NVec(self.nodeCnt, settings['vecDims'])
            self.rVec       = RVec(self.nodeCnt, settings['vecDims'])

    def forward(self, x):
        positive = copy.copy(x)
        column_random = self.random(0, self.nodeCnt-1, size=len(x))
        negative = x
        
        if np.random.uniform() < 0.5:
            negative[:,0] = column_random
        else:
            negative[:,2] = column_random
        margin_loss = self.distance(positive) - self.distance(negative) + self.margin
        
        self.loss       = F.sum(F.relu(margin_loss))
        # self.accuracy   = self.accuracy_func(x, self.settings['hit'])
        reporter.report({'loss': self.loss}, self)
        # reporter.report({'accuracy': self.accuracy}, self)
        return self.loss.reshape(1)
    
    def distance(self, triples, L1_flag=False):
        hi, ri, ti = triples[:,0], triples[:,1], triples[:,2]
        h = self.nVec(hi)
        r = self.rVec(ri)
        t = self.nVec(ti)
        if L1_flag == True:
            dis = F.sum(F.absolute(h + r - t))
        else:
            a = h + r - t
            dis = F.batch_l2_norm_squared(h + r - t)
        return dis

    def accuracy_func(self, x, ranks):
        return 1


class TransE2(chainer.link.Chain):
    def __init__(self, nodeCnt, settings):
        super(TransE2, self).__init__()
        with self.init_scope():
            self.settings   = settings
            self.margin     = settings['margin']
            self.nodeCnt    = nodeCnt
            self.random     = np.random.randint
            self.nVec       = NVec(self.nodeCnt, settings['vecDims'])
            self.rVec       = RVec(self.nodeCnt, settings['vecDims'])

    def forward(self, x):
        positive = copy.copy(x)
        column_random = self.random(0, self.nodeCnt-1, size=len(x))
        negative = x
        
        if np.random.uniform() < 0.5:
            negative[:,0] = column_random
        else:
            negative[:,2] = column_random
        margin_loss = self.distance(positive) - self.distance(negative) + self.margin
        
        self.loss       = F.sum(F.relu(margin_loss))
        reporter.report({'loss': self.loss}, self)
        return self.loss.reshape(1)
    
    def distance(self, triples, L1_flag=False):
        hi, ri, ti = triples[:,0], triples[:,1], triples[:,2]
        h = self.nVec(hi)
        r = self.rVec(ri)
        t = self.nVec(ti)
        if L1_flag == True:
            dis = F.sum(F.absolute(h + r - t))
        else:
            a = h + r - t
            dis = F.batch_l2_norm_squared(h + r - t)
        return dis

