import numpy as np
from torch.utils.data import DataLoader
from .strategy import Strategy
import pickle
from scipy.spatial.distance import cosine
import sys
import gc
from scipy.linalg import det
from scipy.linalg import pinv as inv
from copy import copy as copy
from copy import deepcopy as deepcopy
import torch
from torch import nn
#import torchfile
from torch.autograd import Variable
#import resnet
#import vgg
import torch.optim as optim
import pdb
from torch.nn import functional as F
import argparse
import torch.nn as nn
from collections import OrderedDict
from scipy import stats
import numpy as np
import scipy.sparse as sp
from itertools import product
from sklearn.base import BaseEstimator, ClusterMixin, TransformerMixin
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances_argmin_min
from sklearn.utils.extmath import row_norms, squared_norm, stable_cumsum
from sklearn.utils.sparsefuncs_fast import assign_rows_csr
from sklearn.utils.sparsefuncs import mean_variance_axis
from sklearn.utils.validation import _num_samples
from sklearn.utils import check_array
from sklearn.utils import gen_batches
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES
from sklearn.metrics.pairwise import rbf_kernel as rbf
#from sklearn.externals.six import string_types
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import pairwise_distances

# kmeans ++ initialization
def init_centers(X, K):
    embs = torch.Tensor(X)
    ind = torch.argmax(torch.norm(embs, 2, 1)).item()
    embs = embs.cuda()
    mu = [embs[ind]]
    indsAll = [ind]
    centInds = [0.] * len(embs)
    cent = 0
    print('#Samps\tTotal Distance')
    while len(mu) < K:
        if len(mu) == 1:
            D2 = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy()
        else:
            newD = torch.cdist(mu[-1].view(1,-1), embs, 2)[0].cpu().numpy()
            for i in range(len(embs)):
                if D2[i] >  newD[i]:
                    centInds[i] = cent
                    D2[i] = newD[i]
        print(str(len(mu)) + '\t' + str(sum(D2)), flush=True)
        if sum(D2) == 0.0: pdb.set_trace()
        D2 = D2.ravel().astype(float)
        Ddist = (D2 ** 2)/ sum(D2 ** 2)
        customDist = stats.rv_discrete(name='custm', values=(np.arange(len(D2)), Ddist))
        ind = customDist.rvs(size=1)[0]
        while ind in indsAll: ind = customDist.rvs(size=1)[0]
        mu.append(embs[ind])
        indsAll.append(ind)
        cent += 1
    return indsAll

"""
class BadgeSampling(Strategy):
    def __init__(self, X, Y, idxs_lb, net, handler, args):
        super(BadgeSampling, self).__init__(X, Y, idxs_lb, net, handler, args)

    def query(self, n):
        idxs_unlabeled = np.arange(self.n_pool)[~self.idxs_lb]
        gradEmbedding = self.get_grad_embedding(self.X[idxs_unlabeled], self.Y.numpy()[idxs_unlabeled]).numpy()
        chosen = init_centers(gradEmbedding, n)
        return idxs_unlabeled[chosen]
"""
from dataloader.paths import PathsDataset
from utils.misc import turn_on_dropout, visualize_entropy
import constants
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm


class BadgeSampling:

    def __init__(self, dataset, lmdb_handle, base_size, batch_size, num_classes, device):
        self.lmdb_handle = lmdb_handle
        self.num_classes = num_classes
        self.base_size = base_size
        self.batch_size = batch_size
        self.dataset = dataset
        self.device = device
        
    def get_grad_embedding(self, data, model=[]):
        if type(model) == list:
            model = self.clf
        dataiter = iter(data)
        single_batch = next(dataiter)['label']
        embDim = model.get_embedding_dim()
        model.eval()
        nLab = len(torch.unique(single_batch))
        print(nLab)
        embedding = np.zeros([len(data), embDim * nLab])
        loader_te = data #DataLoader(self.handler(X, Y, transform=self.args['transformTest']),
                            #shuffle=False, **self.args['loader_te_args'])
        
        with torch.no_grad():
            for idxs, sample in enumerate(loader_te):
                x = sample['image'].cuda(self.device)
                y = sample['label'].cuda(self.device)
            #for x, y, idxs in loader_te:
                x, y = Variable(x.cuda()), Variable(y.cuda())
                cout, out = model(x)
                out = out.data.cpu().numpy()
                batchProbs = F.softmax(cout, dim=1).data.cpu().numpy()
                maxInds = np.argmax(batchProbs,1)
                #print(maxInds)
                print(idxs)
                for j in range(len(y)):
                    for c in range(nLab):
                        embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        #if c == maxInds[j]:
                        #    embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (1 - batchProbs[j][c])
                        #else:
                        #    embedding[idxs[j]][embDim * c : embDim * (c+1)] = deepcopy(out[j]) * (-1 * batchProbs[j][c])
            return torch.Tensor(embedding)
        
    def calculate_scores(self, model, paths):
        model.eval()
        scores = []
        loader = DataLoader(PathsDataset(self.lmdb_handle, self.base_size, paths), batch_size=self.batch_size, shuffle=False, num_workers=0)
        BS = self.get_grad_embedding(loader, model)
        with torch.no_grad():
            for sample in tqdm(loader):
                image_batch = sample['image'].cuda(self.device)
                label_batch = sample['label'].cuda(self.device)
                
                softmax = torch.nn.Softmax2d()
                output = softmax(model(image_batch))
                num_classes = output.shape[1]
                for batch_idx in range(output.shape[0]):
                    entropy_map = torch.cuda.FloatTensor(output.shape[2], output.shape[3]).fill_(0).cuda(self.device)
                    for c in range(self.num_classes):
                        entropy_map = entropy_map - (output[batch_idx, c, :, :] * torch.log2(output[batch_idx, c, :, :] + 1e-12))
                    #entropy_map[label_batch[batch_idx, :, :] == 255] = 0
                    entropy_map[label_batch[batch_idx, :, :] == 0] = 0
                    scores.append(entropy_map.mean().cpu().item())
                    del entropy_map
                torch.cuda.empty_cache()
        return scores

    def ranking(self, model, training_set):

        scores = self.calculate_scores(model, training_set.remaining_image_paths)
        selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1]
        model.eval()
        return selected_samples

    
    def grad_embading(self, paths):
        loader = DataLoader(PathsDataset(self.lmdb_handle, self.base_size, paths), batch_size=self.batch_size, shuffle=False, num_workers=0)
        for sample in tqdm(loader):
            image_batch = sample['image'].cuda(self.device)
            label_batch = sample['label'].cuda(self.device)
        return  image_batch
        
        
    def select_next_batch(self, model, training_set, selection_count):
        X_remaining  = training_set.remaining_image_paths
        gr_emb = self.grad_embading(X_remaining)
        print(gr_emb)
        scores = self.calculate_scores(model, training_set.remaining_image_paths)
        selected_samples = list(zip(*sorted(zip(scores, training_set.remaining_image_paths), key=lambda x: x[0], reverse=True)))[1][:selection_count]
        model.eval()
        training_set.expand_training_set(selected_samples)

