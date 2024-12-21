import numpy as np
import torch
from utils import load_data, set_params_large, clustering_metrics

from utils.preprocess import *
import warnings
import datetime
import time
import random
from kmeans_pytorch import kmeans
from torch.utils.data import RandomSampler
import argparse
from scipy import linalg, sparse
from sklearn.utils.extmath import  safe_sparse_dot
from sklearn.cluster import KMeans
from utils.utils_large import *
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--load_parameters', default=True)
parser.add_argument('--dataset', type=str, default="mag")
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--seed', type=int, default=0)
parser.add_argument('--hidden_dim', type=int, default=256)
parser.add_argument('--embed_dim', type=int, default=128)
parser.add_argument('--nb_epochs', type=int, default=8)
parser.add_argument('--nlayer', type=int, default=2)

parser.add_argument('--l2_coef', type=float, default=1e-4)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--dropout', type=float, default=0.)
parser.add_argument('--tau', type=float, default=1.0)

# model-specific parameters
parser.add_argument('--alpha', type=float, default=0.8)
parser.add_argument('--sigma', type=float, default=0.8)
parser.add_argument('--n_T', type=int, default=10)
parser.add_argument('--n_T2', type=int, default=10)
parser.add_argument('--n_time', type=int, default=10)
parser.add_argument('--fusion', type=lambda x: (str(x).lower() == 'true'), default=False)
parser.add_argument('--gamma', type=float, default=0.1)
parser.add_argument('--dim', type=int, default=64)
parser.add_argument('--lamda', type=float, default=1.0)
parser.add_argument('--theta', type=int, default=20)
parser.add_argument('--round', type=int, default=5)
parser.add_argument('--beta', type=float, default=0.00001)
parser.add_argument('--max_h', type=int, default=1)
args, _ = parser.parse_known_args()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

def train():
    feat, adjs, label = load_data(args.dataset)
    nb_classes = label.shape[-1]
    num_target_node = len(feat)

    feats_dim = feat.shape[1]
    sub_num = int(len(adjs))
    print("Dataset: ", args.dataset)
    print("The number of meta-paths: ", sub_num)
    print("Number of target nodes:", num_target_node)
    print("The dim of target' nodes' feature: ", feats_dim)
    print("Label: ", label.sum(dim=0))
    print(args)

    if torch.cuda.is_available():
        print(f'Using CUDA on {device}')
        adjs = [adj.to(device) for adj in adjs]
        feat = feat.to(device)

    adjs_o,_ = graph_process_large(adjs, feat, args)
    feat = process_feature(feat, args)
    start_time = time.time()
    H = consensus(adjs_o, feat, args)
    con_time=time.time() - start_time
    print("consensus time: ", con_time)
    label = torch.argmax(label, dim=-1)
    label = label.cpu().numpy()
    Q = H
    Q = pcc_norm(Q)
    pcc_time=time.time()-start_time-con_time
    print("pcc_time: ", pcc_time)
    # Q = similarity_learn(Q, adjs_o, args, w_list)

    Q = process_signal(Q, args)
    Q = Q.cpu().numpy()
    process_time = time.time() -con_time-pcc_time-start_time
    print("process_time: ", process_time)
    from sklearn.metrics import normalized_mutual_info_score as nmi
    from sklearn.metrics import adjusted_rand_score as ari
    end_time = time.time() - start_time
    kmeans = KMeans(n_clusters=nb_classes, random_state=42)
    kmeans.fit(Q)
    pre_labels = kmeans.labels_
    acc = clustering_accuracy(label, pre_labels)

    nmi = nmi(label, pre_labels)
    ari = ari(label, pre_labels)
    print("dataset:{},alpha:{},n_T:{},dim:{},theta:{},sigma:{}".format(args.dataset, args.alpha, args.n_T, args.dim,
                                                                       args.theta, args.sigma))
    print("ACC:{:.4},NMI:{:.4},ari:{:.4},time:{:.4}".format(acc, nmi, ari, end_time))
    output_file = '{}_results.txt'.format(args.dataset)

    with open(output_file, 'a') as f:
        f.write("dataset:{}, alpha:{}, n_T:{}, dim:{}\n".format(args.dataset, args.alpha, args.n_T, args.dim))
        f.write("ACC:{:.4}, NMI:{:.4}, ARI:{:.4}, time:{:.4}\n".format(acc, nmi, ari, end_time))



if __name__ == '__main__':

        train()

