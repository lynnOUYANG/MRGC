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
## random seed ##
def set_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

def format_time(time):
    elapsed_rounded = int(round((time)))
    return str(datetime.timedelta(seconds=elapsed_rounded))

def process_feature(X,args):
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
    device = X.device
    X = X.cpu().numpy()

    X = StandardScaler().fit_transform(X)
    pca = PCA(n_components=args.dim,random_state=6)
    X = pca.fit_transform(X)
    # X = square_feat_map(X, 0.00001)
    X = torch.from_numpy(X).to(device).to(torch.float32)

    return X
def F_norm(A):
    if A is sparse:
        values=A._values()
        sum=torch.sum(values**2)
    else:
        sum=torch.sum(A.sum(1)**2)
    F=torch.sqrt(sum)

    # F=torch.norm(A,p='fro')
    return F
def l_norm(A,n):
    norms = torch.norm(A, p=n, dim=1, keepdim=True)
    return A/norms
def norm_w(w_lsit):
    sum=torch.norm(w_lsit,p=1)
    w_norm=w_lsit/sum
    return w_norm
def sinkhorn_knopp_adjust(Z):
    Zl = Z.clone()

    for _ in range(10):
        c = torch.matmul(Zl, torch.sum(Z.T, dim=1))
        c = 1.0 / c

        indices = torch.arange(len(c), device=c.device).unsqueeze(0).repeat(2, 1)
        values = c.flatten()
        diag_sparse = torch.sparse.FloatTensor(indices, values, torch.Size([len(c), len(c)]))

        Zl = torch.sparse.mm(diag_sparse, Zl)
        c = torch.matmul(torch.sum(Zl, dim=0), Z.T)
        c=1.0/c
        values = c.flatten()
        diag_sparse = torch.sparse.FloatTensor(indices, values, torch.Size([len(c), len(c)]))
        Z = torch.sparse.mm(diag_sparse, Z)

    return Z
def process_signal(H,args):
    # H=H*np.sqrt(mean_degree.cpu().numpy())[:, np.newaxis]
    if args.gamma!=0:
        orf = ORF(n_components=H.shape[1], gamma=args.gamma, distribution='gaussian', random_fourier=True,use_offset=False, random_state=42)
    else:
        orf = ORF(n_components=H.shape[1], gamma='auto', distribution='gaussian', random_fourier=True,
                  use_offset=False, random_state=42)
    orf.fit(H,seed=args.seed)
    H = orf.transform(H)

    H=sinkhorn_knopp_adjust(H)

    return H
def filter_double_para(adj,feature,n_T,alpha,sigma):
    para1=alpha/(alpha+sigma)
    para2=sigma/(alpha+sigma)
    feature0=feature
    for i in range(n_T):
        feature=para2*safe_sparse_dot(adj,feature)+feature0
        if i==(n_T-2):
            minus_A=feature

    remainder=feature-minus_A
    H=para2*remainder+para1*feature
    return H
def pcc_norm(A):
    mean_row=A.mean( dim=1, keepdim=True)
    A_norm = A - mean_row
    A_l2=torch.norm(A, p=2, dim=1, keepdim=True)
    A=A_norm/A_l2
    A=torch.nan_to_num(A)
    return A
def consensus(adjs,H,args):
    num_view = len(adjs)
    num_edges=0

    for i in range(len(adjs)):
        num_edges+=adjs[i]._nnz()
    # print(torch.ceil(torch.tensor(args.theta*num_edges,device=H.device)).int())
    device=H.device
    omega=torch.full((len(adjs),), 1.0/np.sqrt(num_view), device=adjs[0].device)
    # print(adjs)

    empty_indices = torch.empty((2, 0), dtype=torch.long)
    empty_values = torch.empty((0,), dtype=torch.float32)
    shape = (H.shape[0], H.shape[0])
    # Ac0= torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)

    H0=H.clone()
    ade=ADE(theta=args.theta,max_h=args.max_h,beta=args.beta)
    # rsc = RSC(3000, m=0.5, laplacian=0, n_iter=1)
    A = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
    Ac_sum = torch.sparse_coo_tensor(empty_indices, empty_values, shape, device=device)
    for j in range(num_view):
        weighted_adj = adjs[j] * (omega[j] ** 2)
        A = A + weighted_adj
    indices = A._indices()
    deg = torch.bincount(indices[0], minlength=A.size(0))
    values, deg_indice = torch.topk(deg, args.theta*10, largest=False)
    top_nodes=torch.stack((values,deg_indice), dim=1)

    for i in range(args.n_time):
        # A = torch.zeros((H.shape[0], H.shape[0]), device=device)
        H = H0
        for j in range(num_view):
            omega[j]=1.0/(F_norm(A-adjs[j])**2)
        omega=norm_w(omega)
        A= Ac_sum
        for j in range(num_view):
            weighted_adj=adjs[j]*(omega[j]**2)
            A=A+weighted_adj
        # A_dense=A.to_dense()
        # A=A_dense.to_sparse()

        # A=A/len(adjs)
        # A=A-Ac0

        # A=A.coalesce()
        # A=merge_duplicate_edges(A)
        # A_dense=A
        # A=A.to_sparse()
        # print(A)
        A,Ac=ade._add_multi_edge(A,H,device,top_nodes)
        add_indice=Ac._indices()[0]

        add_deg=torch.bincount(add_indice, minlength=Ac.size(0))
        add_deg=add_deg[deg_indice]
        top_nodes_values=add_deg+top_nodes[:,0]
        top_nodes = torch.stack((top_nodes_values, deg_indice), dim=1)

        Ac_sum+=Ac
        # print(A)
        # norm_A = normalize_adj_from_tensor(A, 'sym', True)
        # H=H0
        # H = filter_try(norm_A, H, args.n_T, args.alpha)

        # print(A)
        #
        # # Ac0=Ac0+Ac
        # adjs,adjss=update_adj(adjs,Ac)


        norm_A = normalize_adj_from_tensor(A, 'sym', True)


        H = filter_double_para(norm_A, H, args.n_T, args.alpha,args.sigma)
        # H = filter_try(norm_A, H, args.n_T, args.alpha)
        H = l_norm(H, 2)

    # A=A.to_sparse()

    # A=A+adj_I.to_sparse()
    # for i in range(args.round):
    #     H=H0
        # norm_A = normalize_adj_from_tensor(A, 'sym', True)
        # H = filter_try(norm_A, H, args.n_T, args.alpha)
        # H = filter_try(A, H, args.n_T, args.alpha)
        # H=l_norm(H,2)
        # A, Ac = rsc._latent_decomposition(A.to_dense(),A, H0, device)

        # print(A)
        # H = l_norm(H, 2)

    return H