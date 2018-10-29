import numpy as np
import sys, os, time, datetime, shutil
import chainer
from multiprocessing import Pool
from chainer import reporter, Variable, Chain, Parameter, link
from chainer import datasets, iterators, optimizers, training
from chainer.initializers.normal import Normal
from chainer.training import extensions
import chainer.functions as F
import chainer.links as L
import chainer.computational_graph as C
import json, inspect
import cupy
from chainer import serializers
from graph import *
from datahelper import *
from scipy.spatial.distance import cosine

def _check_index(pre, suc):
    assert(len(pre)==len(suc))
    for i in range(len(pre)):
        assert(len(pre[i])>0 or len(suc[i])>0)

def copy_file(src, dst):
    shutil.copy(src, dst)  

class RVec(link.Link):
    def __init__(self, counts, vecDims):
        super(RVec, self).__init__()
        with self.init_scope():
            initializer     = Normal(0.1)
            self.vecDims    = vecDims
            self.edge2vec   = Parameter(initializer)
            self.edge2vec.initialize((counts, vecDims))
    def forward(self, indexs):
        vecs = F.embed_id(indexs, self.edge2vec).reshape(-1, self.vecDims)
        return vecs
        
class NVec(link.Link):
    def __init__(self, counts, vecDims):
        super(NVec, self).__init__()
        with self.init_scope():
            initializer     = Normal(0.1)
            self.counts     = counts
            self.vecDims    = vecDims
            self.node2vec   = Parameter(initializer)
            self._initialize_params()
    def _initialize_params(self):
        self.node2vec.initialize((self.counts, self.vecDims))
    def forward(self, indexs):
        vecs = F.embed_id(indexs, self.node2vec).reshape(-1, self.vecDims)
        return vecs

class Trans(chainer.link.Chain):
    def __init__(self, hidden_unit, output_unit, dropout, linear=False):
        super(Trans, self).__init__()
        self.dropout = dropout
        with self.init_scope():
            self.l1 = L.Linear(200, hidden_unit)
            self.l2 = L.Linear(hidden_unit, output_unit)
            self.linear = linear
    def __call__(self, x):
        if self.linear == True:
            h1 = self.l1(x)
            return self.l2(h1)
        else:
            x1 = F.dropout(x, ratio=0.5)
            h1 = F.relu(self.l1(x))
            x2 = F.dropout(h1, ratio=0.5)
        return self.l2(h1)

# F : Through a nerual network
class Merger(chainer.link.Chain):
    def __init__(self, dim, dropout_rate, activate, isR, isBN):
        super(Merger, self).__init__()
        with self.init_scope():
            self.is_residual    = isR
            self.is_batchnorm   = isBN
            self.activate       = activate
            self.dim            = dim
            self.dropout_rate   = dropout_rate
            self.x2z	        = L.Linear(dim,dim)
            self.bn	            = L.BatchNormalization(dim)
    def forward(self, x, index_array):
        x = x.reshape(-1, self.dim)
        if self.dropout_rate!=0:
            x = F.dropout(x,ratio=self.dropout_rate)
        z = self.x2z(x)
        if self.activate=='tanh':
            z = F.tanh(z)
        if self.activate=='relu':
            z = F.relu(z)
        if self.is_residual:
            z = z+x
        split_array = F.split_axis(z, index_array, axis=0)[:-1]
        a = []
        for i in split_array:
            if len(i)>0:
                a.append(F.average(i,axis=0))
            else:
                a.append(Variable(np.zeros(self.dim, dtype=np.float32)))
        p = F.stack(a)
        return p

# F : only average
class Merger2(chainer.link.Chain):
    def __init__(self):
        super(Merger2, self).__init__()
        with self.init_scope():
            self.pool           = Pool(8)
    def forward(self, x, index_array):
        split_array = F.split_axis(x, index_array, axis=0)[:-1]
        a = []
        for i in split_array:
            if len(i)>0:
                a.append(F.average(i,axis=0))
            else:
                a.append(Variable(np.zeros(self.dim, dtype=np.float32)))
        p = F.stack(a   )
        return p

class GNN(chainer.link.Chain):
    def __init__(self, graph_mes, settings):
        super(GNN, self).__init__()
        with self.init_scope():
            # Settings
            self.settings = settings
            self.pooling    = F.average
            self.Tau        = settings["Tau"]
            self.CPUs       = settings["CPUs"]
            self.vecDims    = settings['vecDims']
            self.dropout    = settings["dropout"]
            self.batch_size = settings["batch_size"]
            self.self_ratio = settings["self_ratio"]
            self.PTrans     = Trans(settings["hiddenUnits_P"], 100, self.dropout, linear=settings['linear_PS'])
            self.STrans     = Trans(settings["hiddenUnits_S"], 100, self.dropout, linear=settings['linear_PS'])
            self.CTrans     = Trans(settings["hiddenUnits_C"], 2,   self.dropout, linear=settings['linear_C'])
            # graph mes
            self.mes = graph_mes
            self.mes.tensor()
            # dataset setting
            self.nvec      = NVec(self.mes.nodeCnt, self.vecDims)
            self.rvec      = RVec(self.mes.edgeCnt, self.vecDims)
            # model modules
            # self.merger     = Merger(self.vecDims, self.dropout, 'relu', isR=True, isBN=True)
            self.merger     = Merger2()
            
    def forward(self, x):
        # negative samples
        h = np.random.randint(0, self.mes.nodeCnt, (x.shape[0],1))
        t = np.random.randint(0, self.mes.nodeCnt, (x.shape[0],1))
        r = x[:,1].reshape(-1,1)
        x_negative = np.concatenate((h,r,t), axis=1)
        # loss 
        loss_x  = self.transloss(x)
        loss_ne = self.transloss(x_negative)
        loss    = F.add(loss_x, F.relu(loss_ne - self.Tau))
        self.loss   = F.sum(loss, axis=0).reshape(1) # loss的值必须是一个variable的列表
        reporter.report({'loss': self.loss[0]}, self)
        accuracy = self.accuracy_func(x, self.settings['hit'])
        reporter.report({'accuracy': accuracy[0]}, self)
        return self.loss

    def accuracy_func(self, x, ranks):
        x = np.array([i for i in x])
        h = x[:,0]
        r = x[:,1]
        t = x[:,2]
        h_vecs = self.nvec(h)
        t_vecs = self.nvec(t)
        pred_vecs       = t_vecs - h_vecs
        pred_vecs       = pred_vecs.reshape(-1,self.vecDims).data
        pred_vecs_l2    = np.linalg.norm(pred_vecs.data, axis=1).reshape(-1,1)
        all_vecs_l2     = np.linalg.norm(self.rvec.edge2vec.data, axis=1).reshape(-1,1)
        norms           = np.dot(pred_vecs_l2,all_vecs_l2.T)
        inners          = np.dot(pred_vecs,self.rvec.edge2vec.data.T)
        cosines         = 1 - inners/norms
        
        real_cosine = np.array([cosines[i][r[i]] for i in range(len(cosines))])
        real_cosines = np.array([np.sort(cosines[i].data) for i in range(len(cosines))])
        real_cosines_hit_last = real_cosines[:,ranks-1]
        accuarcy = [real_cosine[i]<=real_cosines_hit_last[i] for i in range(len(real_cosines))]
        accuarcy = np.float32(sum(accuarcy)/len(real_cosines)).reshape(1)
        return Variable(accuarcy)

    def transloss(self, x):
        hdx     = x[:,0]
        rdx     = x[:,1]
        tdx     = x[:,2]
        hvec    = self.propagation(hdx, "1")
        tvec    = self.propagation(tdx, "2")
        rvec    = self.rvec(rdx)
        return F.batch_l2_norm_squared(hvec+rvec-tvec)
        
        
    def propagation(self, idxs, graph):
        mes = self.mes.tensor2(idxs)
        ivecs = self.nvec(idxs)
        
        pre_mes = mes[0][0]
        suc_mes = mes[1][0]
        pre_pre_mes, pre_suc_mes = mes[0][1]
        suc_pre_mes, suc_suc_mes = mes[1][1]

        pre_dot = self.dot(pre_pre_mes, pre_suc_mes, graph)
        suc_dot = self.dot(suc_pre_mes, suc_suc_mes, graph)
        pre_r_vecs  = self.rvec(pre_mes.links)
        suc_r_vecs  = self.rvec(suc_mes.links)
        pre_concat  = F.concat((pre_dot,pre_r_vecs),axis=1)
        pre_dot2    = self.PTrans(pre_concat)
        pre_dot2    = self.self_ratio*pre_dot2 + (1-self.self_ratio)*pre_r_vecs
        suc_concat  = F.concat((suc_dot,suc_r_vecs),axis=1)
        suc_dot2    = self.STrans(suc_concat)
        suc_dot2    = self.self_ratio*suc_dot2 + (1-self.self_ratio)*suc_r_vecs
        
        if pre_dot is not None and suc_dot is not None:
            pre_dot_group = F.split_axis(pre_dot2, pre_mes.indexs, axis=0)[:-1]
            suc_dot_group = F.split_axis(suc_dot2, suc_mes.indexs, axis=0)[:-1]
        elif pre_dot is not None:
            pre_dot_group = F.split_axis(pre_dot2, pre_mes.indexs, axis=0)[:-1]
            suc_dot_group = None
        elif suc_dot is not None:
            suc_dot_group = F.split_axis(suc_dot2, suc_mes.indexs, axis=0)[:-1]
            pre_dot_group = None
        dot = self.caculate(pre_dot_group, suc_dot_group)
        dot = self.self_ratio*ivecs + (1-self.self_ratio)*dot

        return dot
        
    def dot(self, pre_mes, suc_mes, graph):
        if suc_mes.nodes == [] and pre_mes.nodes == []:
            return None
        if pre_mes.nodes != []:
            pre_n_vecs      = self.nvec(pre_mes.nodes)
            pre_r_vecs      = self.rvec(pre_mes.links)
            pre_concat      = F.concat((pre_n_vecs,pre_r_vecs),axis=1)
            pre_dot         = self.PTrans(pre_concat)
            pre_dot_group   = F.split_axis(pre_dot, pre_mes.indexs, axis=0)[:-1]
        else:
            pre_dot_group = None
        if suc_mes.nodes != []:
            suc_n_vecs      = self.nvec(suc_mes.nodes)
            suc_r_vecs      = self.rvec(suc_mes.links)
            suc_concat      = F.concat((suc_n_vecs,suc_r_vecs),axis=1)
            suc_dot         = self.STrans(suc_concat)
            suc_dot_group   = F.split_axis(suc_dot, suc_mes.indexs, axis=0)[:-1]
        else:
            suc_dot_group = None

        dot = self.caculate(pre_dot_group, suc_dot_group)
        return dot

    def caculate(self, pre, suc):
        arrays = []
        arrays_index = []
        index = 0
        if pre!=None and suc!=None:
            for i in range(len(pre)):
                array = F.concat((pre[i],suc[i]), axis=0)                
                arrays.append(array)
                index += array.shape[0]
                arrays_index.append(index)
        elif pre==None and suc!=None:
            for i in range(len(suc)):
                arrays.append(suc[i])
                index += suc[i].shape[0]
                arrays_index.append(index)
        elif suc==None and pre!=None:
            for i in range(len(pre)):
                arrays.append(pre[i])
                index += pre[i].shape[0]
                arrays_index.append(index)
        else:
            return None
        arrays = F.concat(arrays, axis=0)
        arrays_index = np.array(arrays_index)
        dot_array = self._caculate(arrays, arrays_index)
        return dot_array

    def _caculate(self, arrays, arrays_index):
        if len(arrays) > 0:
            result = self.merger(arrays, arrays_index)
        else:
            return Variable(np.zeros(self.vecDims, dtype=np.float32))
        return result
        
def train(overlap_rate, train_rate):
    settings = Settings()
    sf = settings["file"]
    sf_id = sf[:-4]+"_id"+sf[-4:]
    graph_mes = GraphMes(file=sf_id, ints=True)
    trainset, testset = graph_mes.dataset(train_rate)
    
    train_iterator  = chainer.iterators.SerialIterator(trainset, settings['batch_size'], repeat=True)
    test_iterator  = chainer.iterators.SerialIterator(testset, settings['batch_size'], repeat=False)
    gnn = GNN(graph_mes, settings)

    optimizer = optimizers.Adam(alpha=settings['alpha'],beta1=settings['beta1'],beta2=settings['beta1'],eps=settings['eps'])
    optimizer.setup(gnn)
    updater = training.updater.StandardUpdater(train_iterator,optimizer)

    dt_now = datetime.datetime.now()
    time_stample = './result/result'+str(dt_now.day)+'_'+str(dt_now.hour)+'_'+str(dt_now.minute)+'_'+str(dt_now.second)+'/'
    os.mkdir(time_stample)
    copy_file('../model/settings.json', time_stample+'settings.json') 
    trainer = training.Trainer(updater,(settings['epoch'], 'epoch'), out=time_stample)

    trainer.extend(extensions.Evaluator(test_iterator, gnn))
    trainer.extend(extensions.LogReport())
    
    trainer.extend(extensions.PrintReport(['epoch', 'main/loss', 'validation/main/loss',
         'main/accuracy', 'validation/main/accuracy', 'elapsed_time']))
    trainer.extend(extensions.PlotReport(['main/loss', 'validation/main/loss'],'epoch', file_name='loss.png'))
    trainer.extend(extensions.PlotReport(['main/accuracy', 'validation/main/accuracy'],'epoch', file_name='accuracy.png'))
    trainer.extend(extensions.ProgressBar())
    with cupy.cuda.Device(3):
        trainer.run()

    # serializers.save_npz("mymodel.npz", gnn)

if __name__ == "__main__":
    settings = Settings()
    print("batch_size:"         , settings['batch_size'])
    print("epoch: "             , settings['epoch'])
    print("alpha: "             , settings['alpha'])
    print("beta1: "             , settings['beta1'])
    print("beta2: "             , settings['beta2'])
    print("eps: "               , settings['eps'])
    print("vecDims: "           , settings['vecDims'])
    print("is_parallel: "       , settings['is_parallel'])
    print("overlap_rate: "      , settings['overlap_rate'])
    print("train_rate: "        , settings['train_rate'])
    print("dropout: "           , settings['dropout'])
    print("dropout_vec: "       , settings['dropout_vec'])
    print("hiddenUnits_P: "     , settings['hiddenUnits_P'])
    print("hiddenUnits_S: "     , settings['hiddenUnits_S'])
    print("hiddenUnits_C: "     , settings['hiddenUnits_C'])
    
    # settings
    for overlap_rate in settings['overlap_rate']:
        for train_rate in settings['train_rate']:
            y = train(overlap_rate, train_rate)
