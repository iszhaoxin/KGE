import sys, random, os
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np
from datahelper import *
import matplotlib.pyplot as plt
import json
from mylib.texthelper.format import pshape, pdata, ptree

def Settings():
    with open('../model/settings.json') as f:
        settings = json.load(f)
    return settings

class GraphMes:
    def __init__(self, graph=None, file=None, start=0, ints=False):
        self.helper = DataHelper(file,NP=False)
        self.samples = self.helper.GetSamples()
        if ints == True:
            samples = []
            for i in self.samples:
                samples.append([int(i[0]), int(i[1]),int(i[2])])
            self.samples = np.array(samples)
        if graph==None and file!=None:
            self.G = self.readGraph(file, ints=ints)
        elif graph!=None and file==None:
            self.G = graph
        else:
            raise Exception
        self.start = start
        self.node2id, self.id2node = self._node2id()
        self.edge2id, self.id2edge = self._edge2id()

    def readGraph(self, sf, ints=False):
        self.SamplesCnt = len(self.samples)
        G  = nx.MultiDiGraph()
        for sample in self.samples:
            G.add_edge(sample[0],sample[2],attr=sample[1])
        return G

    def graph2id(self, of):    
        with open(of, 'w') as f:
            for h,r,t in self.samples:
                f.write(str(self.node2id[h])+' '+str(self.edge2id[r])+' '+str(self.node2id[t])+'\n')

    def neighbors_zx(self,node):
        return list(self.G.successors(node))+ list(self.G.predecessors(node)) + [node]
    
    def tensor(self):
        pre_indexs, predecessors, predecessors_group = [], [], []
        suc_indexs, successors, successors_group = [], [], []
        index = 0
        nodes = self.G.nodes()
        for node in nodes:
            j = 0
            for i in self.G.successors(node):
                attr = self.G[node][i][0]['attr']
                successors.append([i,attr])
                j+=1
            index = index+j
            suc_indexs.append(index)
        # 这里就是将扁平化的邻接信息按照node进行分割
        successors_group = np.split(successors,suc_indexs, axis=0)[:-1]
        
        index = 0   
        for node in nodes:
            j = 0
            for i in self.G.predecessors(node):
                attr = self.G[i][node][0]['attr']
                predecessors.append([i,attr])
                j+=1
            index = index+j
            pre_indexs.append(index)
        predecessors_group = np.split(predecessors,pre_indexs, axis=0)[:-1]
        self._check_index(predecessors_group, successors_group)
        print("'\n'len(nodes):",len(nodes))
        print("len(successors_group):",len(successors_group))
        print("len(successors):",len(successors))
        self.predecessors_group = predecessors_group
        self.successors_group = successors_group
    
    def tensor2(self, nodes):
        pre_indexs, predecessors = [], []
        suc_indexs, successors = [], []
        index = 0
        for node in nodes:
            j = 0
            for i in self.G.successors(node):
                attr = self.G[node][i][0]['attr']
                successors.append([i,attr])
                j+=1
            index = index+j
            suc_indexs.append(index)
        # 这里就是将扁平化的邻接信息按照node进行分割
        successors_group = np.split(successors,suc_indexs, axis=0)[:-1]
        
        index = 0   
        for node in nodes:
            j = 0
            for i in self.G.predecessors(node):
                attr = self.G[i][node][0]['attr']
                predecessors.append([i,attr])
                j+=1
            index = index+j
            pre_indexs.append(index)
        predecessors_group = np.split(predecessors,pre_indexs, axis=0)[:-1]
        suc_mes = Adjacency(successors, suc_indexs)
        pre_mes = Adjacency(predecessors, pre_indexs)
        self._check_index(predecessors_group, successors_group)

        suc_suc_indexss, suc_pre_indexss, pre_pre_indexss, pre_suc_indexss = [], [], [], []
        suc_successorss, suc_predecessorss, pre_successorss, pre_predecessorss = [], [], [], []
        for i in successors_group:
            for j in i:
                for k in self.successors_group[j[0]]:
                    suc_successorss.append(k)
                for k in self.predecessors_group[j[0]]:
                    suc_predecessorss.append(k)
                suc_suc_indexss.append(len(suc_successorss))
                suc_pre_indexss.append(len(suc_predecessorss))
        suc_successorss = np.array(suc_successorss)
        suc_predecessorss = np.array(suc_predecessorss)
        suc_pre_mes = Adjacency(suc_predecessorss, suc_pre_indexss)
        suc_suc_mes = Adjacency(suc_successorss, suc_suc_indexss)        
        
        for i in predecessors_group:
            for j in i:
                for k in self.successors_group[j[0]]:
                    pre_successorss.append(k)
                for k in self.predecessors_group[j[0]]:
                    pre_predecessorss.append(k)
                pre_suc_indexss.append(len(pre_successorss))
                pre_pre_indexss.append(len(pre_predecessorss))
        pre_successorss = np.array(pre_successorss)
        pre_predecessorss = np.array(pre_predecessorss)
        pre_pre_mes = Adjacency(pre_predecessorss, pre_pre_indexss)
        pre_suc_mes = Adjacency(pre_successorss, pre_suc_indexss)
        return [[pre_mes,[pre_pre_mes, pre_suc_mes]], [suc_mes,[suc_pre_mes, suc_suc_mes]]]
    
    def _check_index(self, pre, suc):
        assert(len(pre)==len(suc))
        for i in range(len(pre)):
            assert(len(pre[i])>0 or len(suc[i])>0)
        
    def _node2id(self):
        node2id = dict()
        id2node = dict()
        index = 0
        for node in self.G.nodes():
            node2id.update({node:self.start+index})
            id2node.update({self.start+index:node})
            index += 1
        return node2id, id2node
    def _edge2id(self):
        edge2id = dict()
        id2edge = dict()
        self.attrs = set()
        for edge in self.G.edges():
            for i in self.G[edge[0]][edge[1]]:
                # print(self.G[edge[0]][edge[1]][i]['attr'])
                self.attrs.add(self.G[edge[0]][edge[1]][i]['attr'])
        index = 0
        for attr in self.attrs:
            edge2id.update({attr:self.start+index})
            id2edge.update({self.start+index:attr})
            index += 1
        return edge2id, id2edge
    @property
    def nodes(self):
        return list(self.G.nodes)
    @property
    def nodeCnt(self):
        return len(self.G.nodes)
    @property
    def samplesCnt(self):
        return self.SamplesCnt
    @property
    def edges(self):
        return list(self.attrs)
    @property
    def edgeCnt(self):
        return len(self.attrs)
    def id2file(self, nodefn, edgefn):
        with open(nodefn, 'w') as nf:
            for i in range(len(self.node2id)):
                nf.write(self.id2node[i]+' '+str(i)+'\n')
        with open(edgefn, 'w') as ef:
            for i in range(len(self.edge2id)):
                ef.write(self.id2edge[i]+' '+str(i)+'\n')

    def dataset(self, train_rate):
        np.random.shuffle(np.array(self.samples))
        train_index = int(train_rate*len(self.samples))
        
        trainset = np.array(self.samples[:train_index])
        testset = np.array(self.samples[train_index:])
        return trainset, testset
class Adjacency:
    
    def __init__(self,pairs,indexs):
        self.pairs = np.array(pairs)
        self.indexs = indexs
        self.pair_group = np.split(pairs,indexs, axis=0)[:-1]
        if len(pairs)!=0:
            self.nodes  = np.array(self.pairs[:,0])
            self.links  = np.array(self.pairs[:,1])
        else:
            self.nodes  = np.array([])
            self.links  = np.array([])
        
if __name__ == "__main__":
    train_rate=0.5
    settings = Settings()
    sf = settings["file"]
    sf_id = sf[:-4]+"_id"+sf[-4:]
    G = GraphMes(file=sf, ints=False)
    G.graph2id(sf_id)
    print(sf_id)
    G_id = GraphMes(file=sf_id, ints=True)
    # trainset, testset = G_id.dataset(train_rate)
    # G_id.tensor()
    # nodes = np.array([2045])
    # G_id.tensor2(nodes)
