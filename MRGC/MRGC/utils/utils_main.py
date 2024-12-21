import numpy as np
import torch
# from utils import load_data, set_params, clustering_metrics,RSC
# from module.BMGC import *
from utils.preprocess import *
import warnings
import datetime
import random
from sklearn.cluster import KMeans
from scipy import linalg, sparse
from sklearn.utils import check_random_state
from sklearn.utils.extmath import  safe_sparse_dot,svd_flip
from utils.random_feature import OrthogonalRandomFeature as ORF
from utils.ADE import ADE
# def square_feat_map(z, c=1):
#   from sklearn.preprocessing import PolynomialFeatures
#   polf = PolynomialFeatures(include_bias=True)
#   x = polf.fit_transform(z)
#   coefs = np.ones(len(polf.powers_))
#   coefs[0] = c
#   coefs[(polf.powers_ == 1).sum(1) == 2] = np.sqrt(2)
#   coefs[(polf.powers_ == 1).sum(1) == 1] = np.sqrt(2*c)
#   return x * coefs
def F_norm(A):
    if A is sparse:
        values = A._values()
        sum = torch.sum(values ** 2)
    else:
        sum = torch.sum(A.sum(1) ** 2)
    F = torch.sqrt(sum)

    # F=torch.norm(A,p='fro')
    return F


## random seed ##
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def l_norm(A, n):
    norms = torch.norm(A, p=n, dim=1, keepdim=True)
    return A / norms


def pcc_norm(A):
    mean_row = A.mean(dim=1, keepdim=True)
    A_norm = A - mean_row
    A_l2 = torch.norm(A, p=2, dim=1, keepdim=True)
    A = A_norm / A_l2
    A = torch.nan_to_num(A)
    return A


# def weight(adj,args):
#     num_view = len(adj)
#     w_list = torch.full((num_view,), 1 / num_view, device=adj[0].device)
#     adj_sum=torch.empty((adj[0].shape[0],adj[0].shape[1]), device=adj[0].device)
#     for i in range(num_view):
#         adj_sum+=adj[i]
#     adj_sum=adj_sum/num_view
#     for i in range(args.n_time):
#
#         for n in range(num_view):
#             w_list[n] = 1 / (F_norm(adj_sum - adj[n]) **2)
#         w_list = norm_w(w_list)*num_view
#         H0 = torch.zeros((adj[0].shape[0], adj[0].shape[1]), device=adj[0].device)
#         for n in range(num_view):
#             H0 += (w_list[n]**2) * adj[n]
#
#         adj_sum=H0
#     print(adj_sum)
#         # adj_sum = normalize_adj_from_tensor(adj_sum, 'sym', False)
#     # adj_sum=torch.zeros((adj[0].shape[0], adj[0].shape[1]), device=adj[0].device)
#     # for i in range(num_view):
#     #     adj_sum+=w_list[i]*adj[i]
#
#     # adj_sum=w_list[0]*adj[0]+w_list[1]*adj[1]+w_list[2]*adj[2]
#     return adj_sum,w_list
# def update_w(w_list, adj_sum,adj, k, num_view, args):
#     Lf = 0
#     for i in range(num_view):
#         Lf += torch.sum(torch.norm(adj_sum-adj[i], p=1, dim=1, keepdim=True))
#     Lf += 2 * args.lamda
#     R = torch.sqrt(2*torch.log(torch.tensor(num_view, dtype=torch.float32))) / (Lf*torch.sqrt(torch.tensor(k+1, dtype=torch.float32)))
#     for i in range(num_view):
#         f = 2*args.lamda*w_list[i] + torch.sum(torch.norm(adj_sum-adj[i], p=1, dim=1, keepdim=True))
#
#         w_list[i] = w_list[i]*torch.exp(-(R *f))
#     return w_list
# def weight_l2(adj,args):
#     num_view = len(adj)
#     w_list = torch.full((num_view,), 1 / num_view, device=adj[0].device)
#     adj_sum=torch.empty((adj[0].shape[0],adj[0].shape[1]), device=adj[0].device)
#     for i in range(num_view):
#         adj_sum+=adj[i]
#     adj_sum=adj_sum/num_view
#     for i in range(args.n_time):
#
#         # for n in range(num_view):
#             # w_list[n] = 1 / (F_norm(adj_sum - adj[n]) * 2)
#         w_list=update_w(w_list,adj_sum,adj,i,num_view,args)
#         w_list = norm_w(w_list)
#         H0 = torch.empty((adj[0].shape[0], adj[0].shape[1]), device=adj[0].device)
#         for n in range(num_view):
#             H0 += w_list[n] * adj[n]
#         adj_sum = H0
#         print(w_list)
#     # adj_sum=torch.zeros((adj[0].shape[0], adj[0].shape[1]), device=adj[0].device)
#     # for i in range(num_view):
#     #     adj_sum+=w_list[i]*adj[i]
#
#     # adj_sum=w_list[0]*adj[0]+w_list[1]*adj[1]+w_list[2]*adj[2]
#     return adj_sum,w_list
# def filter_w(adj_sum,adj,feature,args):
#
#     omega = torch.zeros((len(adj), feature.shape[0], feature.shape[0]), device=feature.device)
#     # feature_sum0=feature.clone()
#     # feature_sum=feature.clone()
#     # for n in range(4):
#     #     feature_sum=3.5*torch.matmul(adj_sum,feature_sum)+feature_sum0
#     # print(feature_sum)
#     # feature_con=torch.zeros((len(adj), adj[0].shape[1], feature.shape[1]),device=feature.device)
#     # for i in range(len(adj)):
#     #     feature_con[i]=feature.clone()
#     #     feature0=feature.clone()
#     #     for n in range(args.n_T):
#     #         feature_con[i]=args.alpha*safe_sparse_dot(adj[i],feature_con[i])+feature0
#     for n in range(args.n_time):
#         for i in range(len(adj)):
#             # sum=torch.matmul(adj[i].T,adj_sum)
#             sum=safe_sparse_dot(adj[i],adj_sum)
#             U,_,VT=torch.linalg.svd(sum)
#             omega[i]=torch.matmul(U,VT)
#         H=torch.zeros((adj[0].shape[0], adj[0].shape[0]), device=feature.device)
#         for i in range(len(adj)):
#             # H+=torch.matmul(adj[i],omega[i])
#             H+=safe_sparse_dot(adj[i],omega[i])
#         adj_sum=H/len(adj)
#     return adj_sum

# def iter_filter(adj,feature,args):
#     adj0=[]
#     for n in range(len(adj)):
#         adj0.append(adj[n])
#     w_list=torch.empty(len(adj),device=feature.device)
#     adj_zero=torch.empty((len(adj),adj[0].shape[0],adj[0].shape[0]),device=feature.device)
#     adj_sum=torch.zeros((adj[0].shape[0],adj[0].shape[0]),device=feature.device)
#     for n in range(len(adj)):
#         adj_sum+=adj[n]
#     adj_sum=adj_sum/len(adj)
#     for n in range(args.n_time):
#         for n in range(len(adj)):
#             w_list[n] = 1 / (F_norm(adj_sum - adj[n]) * 2)
#         w_list=norm_w(w_list)
#         adj_sum = torch.empty((adj[0].shape[0], adj[0].shape[0]), device=feature.device)
#         for n in range(len(adj)):
#             adj_sum+=w_list[n]*adj[n]
#         adj_sum = adj_sum/len(adj)
#
#         # print(w_list)
#     adj_sum=torch.zeros((adj[0].shape[0], adj[0].shape[1]), device=adj[0].device)
#     for i in range(len(adj)):
#         adj_sum+=w_list[i]*adj[i]
#     # adj_sum=w_list[0]*adj[0]+w_list[1]*adj[1]+w_list[2]*adj[2]
#     # print(adj_sum)
#     feature0=feature.clone()
#     for n in range(args.n_T):
#         feature=args.alpha*safe_sparse_dot(adj_sum,feature)+feature0
#
#     return feature
# def filter(adj,feature,args):
#     feature0 = feature.clone().detach()
#     feature_ori = feature.clone().detach()
#
#     Hr=torch.empty((len(adj),feature.shape[0],feature.shape[1]),device=feature.device)
#
#
#     for n in range(len(adj)):
#         feature=feature_ori.clone()
#         for _ in range(args.n_T2):
#             feature=args.alpha2*(safe_sparse_dot(adj[n],feature)+feature0)
#         Hr[n,:,:]=feature
#     H=Hr.mean(dim=0)
#
#     return H,Hr
# def similarity_learn(H,adj,args,b_list):
#     U=[]
#     H_row=l_norm(H,2)
#     device=H.device
#     for i in range(len(adj)):
#         adj[i]=adj[i].to_dense().cpu().numpy()
#         adj[i]=adj[i]-np.diag(np.sum(adj[i],axis=1))
#         Ur,_,_=randomized_svd(adj[i],n_components=H.shape[1],random_state=42)
#         Ur=torch.from_numpy(Ur).to(device).to(torch.float32)
#         U.append(Ur)
#     U_sum=torch.zeros((U[0].shape[0],U[0].shape[1]),device=device)
#     lam_list = torch.full((2,), 0.5, device=H.device)
#
#     for i in range(len(adj)):
#         U_sum += b_list[i] * U[i]
#     Z=U_sum+H_row*2
#     for i in range(args.n_time):
#         U_sum = torch.zeros((U[0].shape[0], U[0].shape[1]), device=device)
#         for i in range(len(adj)):
#             b_list[i]=1.0/(F_norm(Z-U[i])*2)
#         b_list=norm_w(b_list)
#         for i in range(len(adj)):
#             U_sum += b_list[i] * U[i]
#         Z=lam_list[0]*U_sum+lam_list[1]*H_row
#         lam_list[0]=1.0/(F_norm(Z-U_sum))
#         lam_list[1]=1.0/(F_norm(Z-H_row))
#         lam_list=norm_w(lam_list)
#         print(lam_list)
#         print(b_list)
#
#     return H
def filter_try(adj, feature, n_T, alpha):
    feature0 = feature.clone().detach()

    for _ in range(n_T):
        feature = alpha * (safe_sparse_dot(adj, feature)) + feature0

    return feature


def norm_w(w_lsit):
    sum = torch.norm(w_lsit, p=1)
    w_norm = w_lsit / sum
    return w_norm


# def iter_w(H,Hr,args):
#     num_view=len(Hr)
#     w_list=torch.full((num_view,),1/num_view,device=H.device)
#     omga = torch.empty((num_view, H.shape[1], H.shape[1]), device=H.device)
#     for i in range(args.n_time):
#         H0=torch.empty((H.shape[0],H.shape[1]),device=H.device)
#         # for n in range(num_view):
#         #     sum=torch.matmul(Hr[n].T,H)
#         #     U,_,VT=torch.linalg.svd(sum)
#         #     omga[n]=torch.matmul(U,VT)
#         for n in range(num_view):
#             w_list[n]=1/(F_norm(H-Hr[n])*2)
#         w_list=norm_w(w_list)
#         for n in range(num_view):
#             H0+=w_list[n]*Hr[n]
#         H = H0
#         print(w_list)
#     return H


def clustering_accuracy(y_true, y_pred):
    from sklearn.metrics import confusion_matrix
    from scipy.optimize import linear_sum_assignment

    def ordered_confusion_matrix(y_true, y_pred):
        conf_mat = confusion_matrix(y_true, y_pred)
        w = np.max(conf_mat) - conf_mat
        row_ind, col_ind = linear_sum_assignment(w)
        conf_mat = conf_mat[row_ind, :]
        conf_mat = conf_mat[:, col_ind]
        return conf_mat

    conf_mat = ordered_confusion_matrix(y_true, y_pred)
    return np.trace(conf_mat) / np.sum(conf_mat)


# def filter_new(adj,feature,n_T,alpha):
#     feature0=feature.clone().detach()
#
#     for _ in range(n_T):
#         feature=1/(alpha+1)*(safe_sparse_dot(adj,feature))+feature0
#     feature=alpha/(alpha+1)*feature
#     # for _ in range(n_T):
#     #     feature=(safe_sparse_dot(adj,feature))
#     return feature

# def sinkhorn_knopp_adjust(Z):
#     Zl=Z.copy()
#
#     for _ in range(10):
#         c=np.dot(Zl,np.sum(Z.T,axis=1))
#         # c = 1.0 / np.sqrt(c)
#         c = 1.0 / c
#         Zl = np.diag(c.flatten())@Zl
#         c = np.dot(np.sum(Zl,axis=0),Z.T)
#         c = 1.0 / c
#         Z = np.diag(c.flatten())@Z
#
#     return Z


def sinkhorn_knopp_adjust(Z):
    Zl = Z.clone()

    for _ in range(10):
        c = torch.matmul(Zl, torch.sum(Z.T, dim=1))
        # c = 1.0 / torch.sqrt(c)
        c = 1.0 / c
        Zl = torch.matmul(torch.diag(c.flatten()), Zl)
        c = torch.matmul(torch.sum(Zl, dim=0), Z.T)
        c = 1.0 / c
        Z = torch.matmul(torch.diag(c.flatten()), Z)

    return Z


def process_signal(H, args):
    # H=H*np.sqrt(mean_degree.cpu().numpy())[:, np.newaxis]
    if args.gamma != 0:
        orf = ORF(n_components=H.shape[1], gamma=args.gamma, distribution='gaussian', random_fourier=True,
                  use_offset=False, random_state=42)
    else:
        orf = ORF(n_components=H.shape[1], gamma='auto', distribution='gaussian', random_fourier=True,
                  use_offset=False, random_state=42)
    orf.fit(H, seed=args.seed)
    H = orf.transform(H)

    H = sinkhorn_knopp_adjust(H)

    return H


def process_feature(X, args):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    device = X.device
    X = X.cpu().numpy()

    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=args.dim, random_state=6)
    X = pca.fit_transform(X)
    # X = square_feat_map(X, 0.00001)
    X = torch.from_numpy(X).to(device).to(torch.float32)

    return X


# def merge_duplicate_edges(coo_matrix):
#
#     indices = coo_matrix.indices()
#     values = coo_matrix.values()
#     shape = coo_matrix.shape
#
#     indices_tuple = [tuple(index) for index in indices.t().tolist()]
#
#     merged_dict = {}
#     for idx, value in zip(indices_tuple, values):
#         if idx in merged_dict:
#             merged_dict[idx] += value.item()
#         else:
#             merged_dict[idx] = value.item()
#
#     merged_indices = torch.tensor(list(merged_dict.keys()), dtype=torch.long).t()
#     merged_values = torch.tensor(list(merged_dict.values()), dtype=values.dtype)
#
#     merged_coo_matrix = torch.sparse_coo_tensor(merged_indices, merged_values, shape)
#
#     return merged_coo_matrix
# def mask_zero(adjss):
#     for i in range(len(adjss)):
#         indices=adjss[i]._indices()
#         values=adjss[i]._values()
#         print(values.shape)
#         mask = values >= 0
#         new_indices = indices[:, mask]
#         new_values = values[mask]
#         print(new_values.shape)
#
#         A_new = torch.sparse_coo_tensor(new_indices, new_values, adjss[i].size())
#
#         adjss[i] = A_new.coalesce()
#     return adjss
# def update_adj(adjs,Ac):
#     # indices = Ac._indices()
#     # N = Ac.size(0)
#     #
#     # values = torch.ones(indices.size(1), device=Ac.device)
#     #
#     # Ac_new = torch.sparse_coo_tensor(indices, values, (N, N), device=Ac.device)
#     adjs=[(adj-Ac.to_dense()) for adj in adjs]
#     adjss=mask_zero([adj.to_sparse() for adj in adjs])
#     # print(Ac_new)
#     adjs=[adj.to_dense() for adj in adjs]
#
#     # print(adjss)
#     return adjs,adjss
def filter_double_para(adj, feature, n_T, alpha, sigma):
    para1 = alpha / (alpha + sigma)
    para2 = sigma / (alpha + sigma)
    feature0 = feature
    for i in range(n_T):
        feature = para2 * safe_sparse_dot(adj, feature) + feature0
        if i == (n_T - 2):
            minus_A = feature

    remainder = feature - minus_A
    H = para2 * remainder + para1 * feature
    return H


def consensus(adjs, H, args):
    num_view = len(adjs)
    num_edges = 0
    for i in range(len(adjs)):
        num_edges += adjs[i]._nnz()
    # print(torch.ceil(torch.tensor(args.theta*num_edges,device=H.device)).int())
    device = H.device
    omega = torch.full((len(adjs),), 1.0 / np.sqrt(num_view), device=adjs[0].device)
    # print(adjs)

    empty_indices = torch.empty((2, 0), dtype=torch.long)
    empty_values = torch.empty((0,), dtype=torch.float32)
    shape = (H.shape[0], H.shape[0])
    # Ac0= torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)

    H0 = H.clone()
    ade = ADE(theta=args.theta,max_h=args.max_h,beta=args.beta)
    # rsc = RSC(3000, m=0.5, laplacian=0, n_iter=1)
    A = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
    Ac_sum = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
    for j in range(num_view):
        weighted_adj = adjs[j] * (omega[j] ** 2)
        A = A + weighted_adj
    indices = A._indices()
    deg = torch.bincount(indices[0], minlength=A.size(0))
    values, deg_indice = torch.topk(deg, args.theta * 10, largest=False)
    top_nodes = torch.stack((values, deg_indice), dim=1)

    for i in range(args.n_time):
        # A = torch.zeros((H.shape[0], H.shape[0]), device=device)
        H = H0
        for j in range(num_view):
            omega[j] = 1.0 / (F_norm(A - adjs[j]) ** 2)
        omega = norm_w(omega)
        A = Ac_sum
        for j in range(num_view):
            weighted_adj = adjs[j] * (omega[j] ** 2)
            A = A + weighted_adj

        A, Ac = ade._add_multi_edge(A, H, device, top_nodes)
        add_indice = Ac._indices()[0]

        add_deg = torch.bincount(add_indice, minlength=Ac.size(0))
        add_deg = add_deg[deg_indice]
        top_nodes_values = add_deg + top_nodes[:, 0]
        top_nodes = torch.stack((top_nodes_values, deg_indice), dim=1)
        Ac_sum += Ac

        norm_A = normalize_adj_from_tensor(A, 'sym', True)

        H = filter_double_para(norm_A, H, args.n_T, args.alpha, args.sigma)
        # H = filter_try(norm_A, H, args.n_T, args.alpha)
        H = l_norm(H, 2)

    return H

# def enhence_weight(adj,adjs,H,args):
#     device=adjs[0].device
#     num_view = len(adjs)
#     adj_I = torch.eye(H.shape[0], device=device)
#     for i in range(num_view):
#         adj[i]=adj[i]+adj_I
#     degree=[]
#     for i in range(num_view):
#         degree.append(adj[i].sum(dim=1, keepdim=False))
#     con_matrix=adj[0]
#     for i in range(num_view-1):
#         con_matrix=con_matrix*adj[i+1]
#     con_matrix=con_matrix.to_sparse()
#     # con_matrix=remove_diagonal_elements(con_matrix)
#     # print(con_matrix)
#     H0 = H.clone()
#     omega = torch.full((len(adjs),), 1.0 / np.sqrt(num_view), device=adjs[0].device)
#     # empty_indices = torch.empty((2, 0), dtype=torch.long)
#     # empty_values = torch.empty((0,), dtype=torch.float32)
#     # shape = (H.shape[0], H.shape[0])
#     # A = adjs[0] * (omega[0] ** 2)
#     # for j in range(num_view - 1):
#     #     weighted_adj = adjs[j + 1] * (omega[j + 1] ** 2)
#     #     A = A + weighted_adj
#
#     for i in range(args.n_time):
#         # A = torch.zeros((H.shape[0], H.shape[0]), device=device)
#         H = H0
#
#         # A= torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
#         A=adjs[0]*(omega[0]**2)
#         for j in range(num_view-1):
#             weighted_adj=adjs[j+1]*(omega[j+1]**2)
#             A=A+weighted_adj
#
#         A_dense=A.to_dense()
#         A=A_dense.to_sparse()
#
#         for j in range(num_view):
#             omega[j]=1.0/(F_norm(A-adjs[j])**2)
#         omega=norm_w(omega)
#         # norm_A = normalize_adj_from_tensor(A, 'sym', True)
#         degree_sum=degree[0]*omega[0]
#         for j in range(num_view-1):
#             degree_sum+=omega[j+1]*degree[j+1]
#
#         # inv_qrt=1.0/torch.sqrt(degree_sum)
#         #
#         # norm_con_matrix=inv_qrt*con_matrix*inv_qrt
#         norm_con_matrix=normalize_adj_from_tensor(con_matrix, 'sym', True)
#         # norm_con_matrix=remove_diagonal_elements(con_matrix)
#         # norm_A=norm_A+norm_con_matrix*0.1
#
#         # print(A)
#         # print(norm_con_matrix)
#         # exit()
#         A=A+norm_con_matrix*args.theta
#         A_dense = A.to_dense()
#         A = A_dense.to_sparse()
#         # print(A)
#         H = filter_try(A, H, args.n_T, args.alpha)
#         # H = l_norm(H, 2)
#
#     return H