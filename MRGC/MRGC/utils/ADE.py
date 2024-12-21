import time

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import eigsh
from sklearn.neighbors import kneighbors_graph
from sklearn.cluster import k_means
import torch
# import heapq
import utils.myheapq as heapq

def get_smallest_n_elements(tensor, n):

    tensor_list = tensor.tolist()
    smallest_n= heapq.nsmallest(n, tensor_list, key=lambda x: x[0])

    values, indices = zip(*smallest_n)

    values_tensor = torch.tensor(values)
    indices_tensor = torch.tensor(indices,device=tensor.device)

    return values_tensor, indices_tensor


def remap_indices(selected_indices):

    unique_values = selected_indices.unique()
    value_mapping = {val.item(): idx for idx, val in enumerate(unique_values)}
    remapped_indices = torch.tensor([value_mapping[val.item()] for val in selected_indices])
    return remapped_indices

def create_mask_from_sparse(sparse_tensor, deg_indices):

    indices = sparse_tensor._indices()
    values = sparse_tensor._values()

    mask = torch.isin(indices[0], deg_indices).to(sparse_tensor.device)

    selected_indices = indices[:, mask]
    selected_values = values[mask]
    selected_indices_new = remap_indices(selected_indices[0]).to(sparse_tensor.device)

    mask_indice=torch.stack((selected_indices_new,selected_indices[1]))
    new_sparse_tensor = torch.sparse_coo_tensor(mask_indice, torch.ones_like(selected_values,device=sparse_tensor.device),
                                                (len(deg_indices),sparse_tensor.size(0))).to(sparse_tensor.device)

    dense_matrix = new_sparse_tensor.to_dense()

    mask_matrix = (~dense_matrix.bool()).float()

    return mask_matrix
# def mask_H(H,A,i):
#     # for j in range(A.shape[1]):
#     #     if A[i,j]!=0:
#     #         H[j,:]=0
#     new_H=H.clone()
#     # none_zero_indices = (A[i, :] != 0).nonzero(as_tuple=True)[0]
#     #
#     # new_H[none_zero_indices, :] = 0
#     indices = A._indices()
#     values = A._values()
#     row_mask = (indices[0] == i)
#     valid_indices_1 = indices[1][row_mask & (values != 0)]
#     non_zero_indices = valid_indices_1.unique()
#     new_H[non_zero_indices, :] = 0
#     return new_H
class ADE:


    def __init__(self, theta=20,max_h=50,beta=0.1):


        self.theta = theta
        self.max_h = max_h
        self.beta = beta


    def _add_multi_edge(self, A, H, device,top_nodes):
        N = A.shape[0]  # number of nodes
        # print(A)# Calculate node degrees (sum of each row)
        # if N<50000:
        #     A_dense = A.to_dense()
        #     deg = (A_dense != 0).sum(dim=1)
        # else:
        #     indices = A._indices()
        #     deg = torch.bincount(indices[0], minlength=A.size(0))

        # Ag = A.clone().to(device)  # Keep Ag as a sparse tensor

        # Ensure A is on the correct device and in COO format
        if A.device != device:
            A = A.to(device)
        if not A.is_coalesced():
            A = A.coalesce()

        # values,deg_indice=torch.topk(deg,self.theta,largest=False)
        # print(values)
        # print(deg_indice)

        new_row=[]
        new_col=[]
        new_value=[]

        _,deg_indice=get_smallest_n_elements(top_nodes,self.theta)
        H_new = H[deg_indice]
        p=H_new@H.T

        p=p*create_mask_from_sparse(A,deg_indice)
        k=0

        for i in deg_indice:
            # new_H=mask_H(H,A,i)
            # p=H[i,:]@new_H.T
            # print(H,new_H)

            value,j = torch.topk(p[k], self.max_h, largest=True)
            new_row.extend([i.item()] * len(j))
            new_col.extend(j.tolist())
            new_value.extend(value.tolist())
            k+=1
        new_row = torch.tensor(new_row)
        new_col = torch.tensor(new_col)
        new_value = torch.tensor(new_value)*self.beta

        indices = torch.vstack((new_row, new_col))
        add_edges = torch.sparse_coo_tensor(indices, new_value, size=A.size()).to(device)


        # print(add_edges)
        # exit()
        # for it in range(self.n_iter):
            # allowed_to_remove_per_node = torch.ceil(deg * self.m).to(torch.int)

            # Find edges in the lower triangle (ensure edges are on the correct device)
            # edges = torch.nonzero(torch.tril(Ag.to_dense())).t()
            # val=torch.tril(Ag.to_dense()).to_sparse()

            # removed_edges = []
            # p =(torch.norm(H[edges[0]] - H[edges[1]], dim=1)**2)
            # top_p_indices = torch.topk(p, self.theta, largest=True).indices
            # removed_edges = edges[:, top_p_indices]


            # print(removed_edges)
            # exit()
            # for ind in torch.argsort(p, descending=True):
            # for ind in top_p_indices:
            #     e_i, e_j, p_e = edges[0][ind], edges[1][ind], p[ind]
            #
            #     if allowed_to_remove_per_node[e_i] > 0 and allowed_to_remove_per_node[e_j] > 0 and p_e > 0:
            #         allowed_to_remove_per_node[e_i] -= 1
            #         allowed_to_remove_per_node[e_j] -= 1
            #         removed_edges.append((e_i, e_j))
            #         if len(removed_edges) == self.theta:
            #             break

            # if removed_edges is not None:
            #     removed_edges = torch.tensor(removed_edges, dtype=torch.long, device=device)
            #     row_indices, col_indices = removed_edges
            #     values = A_dense[row_indices, col_indices]
            #     Ac = torch.sparse_coo_tensor(removed_edges, values, (N, N), device=device).coalesce()
            #     Ac=Ac+Ac.transpose(0,1)
            #     Ac_dense=Ac.to_dense()
            #     Ag_dense = Ag.to_dense()
            #     Ag_dense = Ag_dense-Ac_dense
            #     # Ag=Ag.to_sparse()
            # else:
            #     Ac = torch.sparse_coo_tensor(([], ([], [])), (N, N), device=device).coalesce()

        if add_edges is not None:
            # removed_edges = torch.tensor(removed_edges, dtype=torch.long, device=device)
            # row_indices, col_indices = removed_edges
            # values = A_dense[row_indices, col_indices]
            # Ac = torch.sparse_coo_tensor(removed_edges, values, (N, N), device=device).coalesce()
            Ac=add_edges
            Ac=Ac+Ac.transpose(0,1)
            # Ac_dense=Ac.to_dense()
            # Ag_dense = Ag.to_dense()
            Ag=A+Ac
            # Ag=Ag.to_sparse()
        else:
            Ac = torch.sparse_coo_tensor(([], ([], [])), (N, N), device=device).coalesce()
            # new_deg = torch.ceil((A_dense != 0).sum(dim=1)).int()
            # mask=new_deg<allowed_to_remove_per_node
            # indices = torch.nonzero(mask, as_tuple=True)[0]
            # # print(indices.shape)
            # for i in range(len(indices)):
            #
            #     node_edges = torch.nonzero(A_dense[indices[i]]).squeeze()
            #     node_edge_indices = edges[:, node_edges]
            #     node_p_values = p[node_edges]
            #
            #     for idx in torch.argsort(node_p_values, descending=True):
            #         e_i, e_j = node_edge_indices[:, idx]
            #         if new_deg[e_i] < allowed_to_remove_per_node[e_i] or new_deg[e_j] < allowed_to_remove_per_node[e_j]:
            #
            #             Ag_dense[e_i, e_j] = A_dense[e_i, e_j]
            #             Ag_dense[e_j, e_i] = A_dense[e_j, e_i]
            #
            #             new_deg[e_i] += 1
            #             new_deg[e_j] += 1
            #         else:
            #             break
            # for i in range(N):
            #     if new_deg[i]<allowed_to_remove_per_node[i]:
                    # print(new_deg[i],allowed_to_remove_per_node[i])
                    # node_edges = torch.nonzero(A_dense[i]).squeeze()
                    # node_edge_indices = edges[:, node_edges]
                    # node_p_values = p[node_edges]
                    #
                    # for idx in torch.argsort(node_p_values, descending=True):
                    #     e_i, e_j = node_edge_indices[:, idx]
                    #     if new_deg[e_i] < allowed_to_remove_per_node[e_i] or new_deg[e_j] < allowed_to_remove_per_node[e_j]:
                    #
                    #         Ag_dense[e_i, e_j] = 1
                    #         Ag_dense[e_j, e_i] = 1
                    #
                    #         new_deg[e_i] += 1
                    #         new_deg[e_j] += 1
                    #     else:
                    #

            # Ag = Ag_dense.to_sparse()

        # print(A,Ag)
        return Ag, Ac

