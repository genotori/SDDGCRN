import csv
import os
import torch
import pickle
import numpy as np
import scipy.sparse as sp
from scipy.sparse import linalg


def sym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).astype(np.float32).todense()


def asym_adj(adj):
    adj = sp.coo_matrix(adj)
    rowsum = np.array(adj.sum(1)).flatten()
    d_inv = np.power(rowsum, -1).flatten()
    d_inv[np.isinf(d_inv)] = 0.
    d_mat = sp.diags(d_inv)
    return d_mat.dot(adj).astype(np.float32).todense()


def calculate_normalized_laplacian(adj):
    adj = sp.coo_matrix(adj)
    d = np.array(adj.sum(1))
    d_inv_sqrt = np.power(d, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    normalized_laplacian = sp.eye(adj.shape[0]) - adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()
    return normalized_laplacian


def calculate_scaled_laplacian(adj_mx, lambda_max=2, undirected=True):
    if undirected:
        adj_mx = np.maximum.reduce([adj_mx, adj_mx.T])
    L = calculate_normalized_laplacian(adj_mx)
    if lambda_max is None:
        lambda_max, _ = linalg.eigsh(L, 1, which='LM')
        lambda_max = lambda_max[0]
    L = sp.csr_matrix(L)
    M, _ = L.shape
    I = sp.identity(M, format='csr', dtype=L.dtype)
    L = (2 / lambda_max * L) - I
    return L.astype(np.float32).todense()


def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


# def load_adj(pkl_filename, adjtype):
#     adj_mx = load_pickle(pkl_filename)
#     if adjtype == "scalap":
#         adj = [calculate_scaled_laplacian(adj_mx)]
#     elif adjtype == "normlap":
#         adj = [calculate_normalized_laplacian(adj_mx).astype(np.float32).todense()]
#     elif adjtype == "symnadj":
#         adj = [sym_adj(adj_mx)]
#     elif adjtype == "transition":
#         adj = [asym_adj(adj_mx)]
#     elif adjtype == "doubletransition":
#         adj = [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
#     elif adjtype == "identity":
#         adj = [np.diag(np.ones(adj_mx.shape[0])).astype(np.float32)]
#     else:
#         error = 0
#         assert error, "adj type not defined"
#     return adj

def pems_load_pickle(pickle_file, node_num, k, zero_one = False):
    with open(pickle_file, 'r') as f:
        reader = csv.reader(f)
        _ = f.__next__()
        edges = []
        for i in reader:
            try:
                edges.append((int(i[0]), int(i[1]), float(i[2])))
            except:
                continue

    A = np.zeros((node_num, node_num), dtype=np.float32)
    cost_std = np.std([each[2] for each in edges], ddof=1)
    for i, j, cost in edges:
        A[i, j] = np.exp(-(cost / cost_std) ** 2)
        A[j, i] = np.exp(-(cost / cost_std) ** 2) # 无向图
    A[A < k] = 0
    if zero_one:
        A[A != 0.] = 1.
    return A

# def pems_load_pickle(pickle_file, node_num, k, zero_one=False, max_distance=400):
#     with open(pickle_file, 'r') as f:
#         reader = csv.reader(f)
#         _ = f.__next__()
#         edges = []
#         for i in reader:
#             try:
#                 edges.append((int(i[0]), int(i[1]), float(i[2])))
#             except:
#                 continue

#     A = np.zeros((node_num, node_num), dtype=np.float32)
#     cost_std = np.std([each[2] for each in edges], ddof=1)
#     # 如果要是根据距离计算，是否考虑扩大范围？
#     for i, j, cost in edges:
#         A[i, j] = cost
#         A[j, i] = cost  # 无向图

#     # 寻找中继节点
#     for k in range(node_num):
#         for i in range(node_num):
#             if A[i, k] == 0 or i == k:  # 不可能通过这个节点和其他节点产生关系
#                 continue
#             for j in range(node_num):
#                 if A[j, k] == 0 or i == j or j == k:  # 不可能通过这个节点和其他节点产生关系
#                     continue
#                 if A[i, j] == 0 and A[i, k] + A[k, j] < max_distance:
#                     A[i][j] = A[i, k] + A[k, j]
#                     # print('成功捕获到中继节点', k, '为', i, '->', j, '更新距离', A[i][j])
#                 elif A[i, j] > 0 and A[i, k] + A[k, j] < A[i, j]:
#                     A[i][j] = A[i, k] + A[k, j]
#                     # print('成功捕获到中继节点', k, '为', i, '->', j, '更新距离', A[i][j])
#     for i in range(node_num):
#         for j in range(node_num):
#             A[i, j] = np.exp(-(A[i, j] / cost_std) ** 2)
#     A[A < k] = 0
#     if zero_one:
#         A[A != 0.] = 1.
#     return A


def pems03_load_pickle(pickle_file, node_num, k, zero_one=False):
    node_set_file = pickle_file[:-3] + 'txt'
    node_idx = 0
    node_d = {}
    with open(node_set_file, 'r') as f:
        for line in f:
            try:
                node_d[int(line.strip())] = node_idx
                node_idx += 1
            except:
                continue

    with open(pickle_file, 'r') as f:
        reader = csv.reader(f)
        _ = f.__next__()
        edges = []
        for i in reader:
            try:
                edges.append((int(i[0]), int(i[1]), float(i[2])))
            except:
                continue

    A = np.zeros((node_num, node_num), dtype=np.float32)
    cost_std = np.std([each[2] for each in edges], ddof=1)
    for i, j, cost in edges:
        A[node_d[i], node_d[j]] = np.exp(-(cost / cost_std) ** 2)
        A[node_d[j], node_d[i]] = np.exp(-(cost / cost_std) ** 2)  # 无向图
    A[A < k] = 0
    if zero_one:
        A[A != 0.] = 1.
    return A


def load_adj(dataset, pkl_filename, node_num, k=0.1, sym_graph=False, zero_one=False):
    dataset = dataset.lower()
    if dataset == "pems03":
        adj_mx = pems03_load_pickle(pkl_filename, node_num, k, zero_one=zero_one)
    else:
        adj_mx = pems_load_pickle(pkl_filename, node_num, k, zero_one=zero_one)
    if sym_graph:
        adj_mx += np.eye(node_num)
        return [adj_mx]
        # return [sym_adj(adj_mx)]
    return [asym_adj(adj_mx), asym_adj(np.transpose(adj_mx))]
