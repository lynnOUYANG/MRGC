import torch
from utils import load_data, clustering_metrics

import warnings
import datetime
from sklearn.cluster import KMeans
import argparse
import time
from utils.utils_main import *
from sklearn.metrics import normalized_mutual_info_score as nmi
from sklearn.metrics import adjusted_rand_score as ari
from utils.preprocess import *
warnings.filterwarnings('ignore')
parser = argparse.ArgumentParser()
parser.add_argument('--load_parameters', default=True)
parser.add_argument('--dataset', type=str, default="acm-3025")
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
parser.add_argument('--dim', type=int, default=256)
parser.add_argument('--lamda', type=float, default=1.0)
parser.add_argument('--theta', type=int, default=20)
parser.add_argument('--round', type=int, default=5)
parser.add_argument('--beta', type=float, default=0.00001)
parser.add_argument('--max_h', type=int, default=3)

args, _ = parser.parse_known_args()

if torch.cuda.is_available():
    device = torch.device("cuda:" + str(args.gpu))
    torch.cuda.set_device(args.gpu)
else:
    device = torch.device("cpu")

def train():

    feat, adjs, label = load_data(args.dataset)
    nb_classes = label.shape[-1]
    # num_target_node = len(feat)
    #
    # feats_dim = feat.shape[1]
    # sub_num = int(len(adjs))
    # print("Dataset: ", args.dataset)
    # print("The number of meta-paths: ", sub_num)
    # print("Number of target nodes:", num_target_node)
    # print("The dim of target' nodes' feature: ", feats_dim)
    # print("Label: ", label.sum(dim=0))
    # print(args)

    device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.is_available():
        print(f'Using CUDA on {device}')
        adjs = [adj.to(device) for adj in adjs]
        feat = feat.to(device)


    # print(degree,mean_degree.shape)
    adjs_o,adjs = graph_process(adjs, feat, args)
    # degree=adjs[0].to_dense().sum(axis=1).to(torch.float32)
    # mean_degree=degree/torch.mean(degree)
    feat=process_feature(feat,args)
    start_time=time.time()

    # Q=iter_filter(adjs_o,feat,args)
    # H,w_list=weight(adjs_o,args)
    # H,w_list=weight_l2(adjs_o,args)
    # H=filter_try(H,feat,args.n_T,args.alpha)
    # H,_=filter(adjs_o,H,args)

    H=consensus(adjs_o,feat,args)
    # H=enhence_weight(adjs,adjs_o,feat,args)
    # H=filter_w(H,adjs_o,feat,args)
    # alpha_list=[0.2,0.4,0.5,0.6,0.7,0.8,0.85,0.9,0.95]
    # T_list=[2,4,6,8,9,10,12]
    label = torch.argmax(label, dim=-1)
    label = label.cpu().numpy()
    # for alpha in alpha_list:
    #     for T in T_list:

    Q=H
    Q = pcc_norm(Q)
    # Q = similarity_learn(Q, adjs_o, args, w_list)

    Q = process_signal(Q,args)
    Q = Q.cpu().numpy()
    end_time=time.time()-start_time
    kmeans = KMeans(n_clusters=nb_classes,random_state=42)
    kmeans.fit(Q)
    pre_labels = kmeans.labels_
    acc=clustering_accuracy(label, pre_labels)

    nmis=nmi(label, pre_labels)
    aris=ari(label, pre_labels)
    print("dataset:{},alpha:{},n_T:{},seed:{},theta:{},sigma:{},gamma:{}".format(args.dataset,args.alpha,args.n_T,args.seed,args.theta,args.sigma,args.gamma))
    print("ACC:{:.4},NMI:{:.4},ari:{:.4},time:{:.4}".format(acc,nmis,aris,end_time))
    output_file = '{}_results.txt'.format(args.dataset)

    with open(output_file, 'a') as f:
        f.write("dataset:{}, alpha:{}, n_T:{}, dim:{}\n".format(args.dataset,args.alpha,args.n_T,args.dim))
        f.write("ACC:{:.4}, NMI:{:.4}, ARI:{:.4}, time:{:.4}\n".format(acc, nmis, aris, end_time))



if __name__:
    train()