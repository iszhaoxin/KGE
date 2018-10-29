import os, re, json
import mylib.texthelper.decorator as decorator
import mylib.texthelper.format as texthelper
import numpy as np
import itertools
import networkx as nx
from scipy.sparse import csr_matrix

Debug = True

class DataHelper:
    def __init__(self, file, NP=False, nodeIndexStart=0, edgeIndexStart=0):
        self.file = file
        self.nodeIndexStart = nodeIndexStart
        self.edgeIndexStart = edgeIndexStart
        self.edges, self.nodes = set(),set()
        self.NP = NP
        self.id()
        self._GraphSet()
        
    def id(self):
        self.node2id = dict()
        self.edge2id = dict()
        nodeIndex = self.nodeIndexStart
        edgeIndex = self.edgeIndexStart
        with open(self.file, 'r') as f:
            for line in f:
                headnode, edge, tailnode = line.split()
                if headnode not in self.node2id.keys():
                    self.node2id.update({headnode:nodeIndex})
                    nodeIndex += 1
                if tailnode not in self.node2id.keys():
                    self.node2id.update({tailnode:nodeIndex})
                    nodeIndex += 1
                if edge not in self.edge2id.keys():
                    self.edge2id.update({edge:edgeIndex})
                    edgeIndex += 1
        self.id2node = {v: k for k, v in self.node2id.items()}
        self.id2edge = {v: k for k, v in self.edge2id.items()}
        # print("nodeIndex:",nodeIndex, "edgeIndex:",edgeIndex, '\n')

    def _GraphSet(self):
        if self.NP == True:
            samples = list(itertools.chain.from_iterable(self.GetSamples())) # 3d->2d
        else:
            samples = self.GetSamples()
        samples = np.array(samples) # list -> np.array
        for triples in samples:
            self.nodes.add(triples[0])
            self.nodes.add(triples[2])
            self.edges.add(triples[1])

    def GetSamples(self):
        triples = []
        with open(self.file, 'r') as tf:
            for line in tf:
                triples.append(line.split())
        triples = np.array(triples)
        if self.NP == True:
            positive_samples = triples[triples[:,3]=="1"][:-1]
            negative_samples = triples[triples[:,3]=="-1"][:-1]
            return positive_samples, negative_samples
        else:
            return triples

    def sampleid2file(self, tf):
        posSamples = self.GetSamples()[0]
        with open(tf, "w") as f:
            for sample in posSamples:
                h = str(self.node2id[sample[0]])
                r = str(self.edge2id[sample[1]])
                t = str(self.node2id[sample[2]])
                f.write(h+' '+r+' '+t+'\n')

    def id2file(self):
        with open(nodefn, 'w') as f:
            for i in texthelper.sortDict(self.node2id, By="value"):
                f.write(i[0]+' '+str(i[1])+'\n')
        with open(edgefn, 'w') as f:
            for i in texthelper.sortDict(self.edge2id, By="value"):
                f.write(i[0]+' '+str(i[1])+'\n')

    @decorator.TimeRecorder
    def tensor(self, debug=Debug):
        nodes_size = len(self.nodes)
        edges_size = len(self.edges)
        print("nodes_size:",nodes_size,"edges_size:",edges_size)
        tensor = []
        triples = self.GetSamples()
        for i in range(edges_size):
            X = np.zeros((nodes_size,nodes_size),dtype=np.int)
            for triple in triples:
                if self.edge2id[triple[1]] == i:
                    X[self.node2id[triple[0]]][self.node2id[triple[2]]] = 1
            tensor.append(csr_matrix(X, dtype=np.int8, shape=(nodes_size, nodes_size)))

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

    def dataset(self):
        np.random.shuffle(np.array(self.samples))
        return self.samples

def toid(train, valid):
    graphmes = GraphMes(file=train)
    of = os.path.dirname(train) + "/train_id.txt"
    nodefn = os.path.dirname(train)+ "/train_entity2id.txt"
    edgefn = os.path.dirname(train)+ "/train_relation2id.txt"
    graphmes.graph2id(of)
    graphmes.id2file(nodefn, edgefn)

    graphmes = GraphMes(file=valid)
    of = os.path.dirname(train) + "/valid_id.txt"
    nodefn = os.path.dirname(train)+ "/valid_entity2id.txt"
    edgefn = os.path.dirname(train)+ "/valid_relation2id.txt"
    graphmes.graph2id(of)
    graphmes.id2file(nodefn, edgefn)

    os.system("cat "+train+" "+valid+" >>"+os.path.dirname(train)+"/middle")
    graphmes = GraphMes(file=os.path.dirname(train)+"/middle")
    cnt = graphmes.nodeCnt
    os.system("echo "+str(cnt)+" >>"+os.path.dirname(train)+"/info")
    os.system("rm "+os.path.dirname(train)+"/middle")
    return(graphmes.nodeCnt)
    
def dataset(train, valid, info):
    train_triples, valid_triples = [], []
    with open(train, 'r') as f:
        for l in f:
            train_triples.append([int(i) for i in l.split()])
    train_triples = np.array(train_triples)

    with open(valid, 'r') as f:
        for l in f:
            valid_triples.append([int(i) for i in l.split()])
    valid_triples = np.array(valid_triples)
    
    np.random.shuffle(np.array(train_triples))
    np.random.shuffle(np.array(valid_triples))
    
    with open(info, 'r') as f:
        info = f.read()
        info = int(info.split()[0])
        
    return train_triples, valid_triples, info

if __name__ == "__main__":
    # train = "../../data/raw/train.txt"
    # valid = "../../data/raw/valid.txt"
    train = "../../data/raw_small/train.txt"
    valid = "../../data/raw_small/valid.txt"
    toid(train, valid)
    # train = "../../data/raw/train_id.txt"
    # valid = "../../data/raw/valid_id.txt"
    # info = "../../data/raw/info"
    train = "../../data/raw_small/train_id.txt"
    valid = "../../data/raw_small/valid_id.txt"
    info = "../../data/raw_small/info"
    train, valid, info = dataset(train, valid, info)
    print(info)

