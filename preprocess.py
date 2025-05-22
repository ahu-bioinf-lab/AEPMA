import numpy as np
import scipy.sparse as sp
import torch
import pandas as pd
import os
import sys
import pickle

def normalize_sym(adj):
    rowsum = np.array(adj.sum(1))
    d_inv_sqrt = np.power(rowsum, -0.5).flatten()
    d_inv_sqrt[np.isinf(d_inv_sqrt)] = 0.
    d_mat_inv_sqrt = sp.diags(d_inv_sqrt)
    return adj.dot(d_mat_inv_sqrt).transpose().dot(d_mat_inv_sqrt).tocoo()

def normalize_row(mx):
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx)
    return mx.tocoo()

def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    indices = torch.from_numpy(np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64))
    values = torch.from_numpy(sparse_mx.data)
    shape = torch.Size(sparse_mx.shape)
    return torch.sparse.FloatTensor(indices, values, shape)

def pretreatment_Pep(prefix):
    pm = pd.read_csv(os.path.join(prefix, "pep-microbe.dat"), encoding='utf-8', delimiter=',',
                     names=['pid', 'mid', 'rating']).reset_index(drop=True)
    pp = pd.read_csv(os.path.join(prefix, "pep-pep.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'p2', 'weight']).reset_index(drop=True)
    mm = pd.read_csv(os.path.join(prefix, "microbe-microbe.dat"), encoding='utf-8', delimiter=',',
                     names=['m1', 'm2', 'weight']).reset_index(drop=True)
    md = pd.read_csv(os.path.join(prefix, "microbe-disease.dat"), encoding='utf-8', delimiter=',',
                     names=['mid', 'did', 'rating']).reset_index(drop=True)
    dd = pd.read_csv(os.path.join(prefix, "disease-disease.dat"), encoding='utf-8', delimiter=',',
                     names=['d1', 'd2', 'weight']).reset_index(drop=True)

    offsets = {'p': 4050, 'm': 4050 + 131}
    offsets['d'] = offsets['m'] + 161
    # * node types
    node_types = np.zeros((offsets['d'],), dtype=np.int32)
    node_types[offsets['p']:offsets['m']] = 1
    node_types[offsets['m']:] = 2

    if not os.path.exists("./preprocessed/node_types.npy"):
        np.save("./preprocessed/node_types", node_types)

    pm_pos = pm[pm['rating'] == 1].to_numpy()[:, :2]
    pm_pos[:, 1] += offsets['p']
    neg_ratings = pm[pm['rating'] == 0].to_numpy()[:, :2]
    neg_ratings[:, 1] += offsets['p']
    assert (pm_pos.shape[0] + neg_ratings.shape[0] == pm.shape[0])

    # * negative pairs
    indices_neg = np.arange(neg_ratings.shape[0])
    np.random.shuffle(indices_neg)
    indices_neg = indices_neg[:pm_pos.shape[0] * 1]
    neg_data = neg_ratings[indices_neg]
    np.savez("./preprocessed/neg_pairs_offset", neg_data=neg_data)

    # * positive pairs
    indices = np.arange(pm_pos.shape[0])
    np.random.shuffle(indices)
    pos_data = pm_pos[indices]
    np.savez("./preprocessed/pos_pairs_offset", pos_data=pos_data)

    adjs_offset = {}
    ## pm
    # pm_npy = pm_pos[indices]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[pm_pos[:, 0], pm_pos[:, 1]] = 1
    adjs_offset['pm'] = sp.coo_matrix(adj_offset)

    # md
    md_npy = md.to_numpy(int)[:, :2]
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[md_npy[:, 0] + offsets['p'], md_npy[:, 1] + offsets['m']] = 1
    adjs_offset['md'] = sp.coo_matrix(adj_offset)

    # pp
    pp_npy = pp.to_numpy(int)[:, :2]
    pp_matrix = np.zeros((4050, 4050), dtype=float)
    pp_score = pp['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(pp_npy[:, 0], pp_npy[:, 1], pp_score):
        adj_offset[i, j] = k
        pp_matrix[i, j] = k
        adj_offset[j, i] = k
        pp_matrix[j, i] = k
    adjs_offset['simpp'] = sp.coo_matrix(adj_offset)

    # mm
    mm_npy = mm.to_numpy(int)[:, :2]
    mm_matrix = np.zeros((131, 131), dtype=float)
    mm_score = mm['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(mm_npy[:, 0] + offsets['p'], mm_npy[:, 1] + offsets['p'], mm_score):
        adj_offset[i, j] = k
        mm_matrix[i - offsets['p'], j - offsets['p']] = k
    adjs_offset['simmm'] = sp.coo_matrix(adj_offset)

    # dd
    dd_npy = dd.to_numpy(int)[:, :2]
    dd_matrix = np.zeros((161, 161), dtype=float)
    dd_score = dd['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(dd_npy[:, 0] + offsets['m'], dd_npy[:, 1] + offsets['m'], dd_score):
        adj_offset[i, j] = k
        dd_matrix[i - offsets['m'], j - offsets['m']] = k
    adjs_offset['simdd'] = sp.coo_matrix(adj_offset)

    np.savez('./preprocessed/combined_matrices.npz', dp_matrix=pp_matrix, pd_matrix=mm_matrix, de_matrix=dd_matrix)
    f2 = open("./preprocessed/adjs_offset.pkl", "wb")
    pickle.dump(adjs_offset, f2)
    f2.close()

def pretreatment_drugmicrobe(prefix):
    # Load the data
    pm = pd.read_csv(os.path.join(prefix, "adj.dat"), encoding='utf-8', delimiter=',',
                     names=['pid', 'mid', 'rating']).reset_index(drop=True)
    pp = pd.read_csv(os.path.join(prefix, "drugsimilarity.dat"), encoding='utf-8', delimiter=',',
                     names=['p1', 'p2', 'weight']).reset_index(drop=True)
    mm = pd.read_csv(os.path.join(prefix, "microbesimilarity.dat"), encoding='utf-8', delimiter=',',
                     names=['m1', 'm2', 'weight']).reset_index(drop=True)
    
    # The original dataset uses 1-based indexing, so we subtract 1 from all indices to convert them to 0-based indexing.
    pm[['pid', 'mid']] = pm[['pid', 'mid']] - 1
    pp[['p1', 'p2']] = pp[['p1', 'p2']] - 1
    mm[['m1', 'm2']] = mm[['m1', 'm2']] - 1
    
    print('==========Step 1 complete==========')

    # Set offsets for node types
    max_pid = pm['pid'].max() + 1
    print(max_pid)
    max_mid = pm['mid'].max() + 1
    print(max_mid)
    
    offsets = {'p': max_pid, 'm': max_pid + max_mid}
    # Initialize node types array
    node_types = np.zeros((offsets['m'],), dtype=np.int32)
    node_types[offsets['p']:] = 1
    
    # Save node types to file
    if not os.path.exists("../preprocessed/node_types.npy"):
        np.save("../preprocessed/node_types", node_types)
    print('==========Step 2 complete==========')

    # Generate positive pairs
    pm_pos = pm[pm['rating'] == 1].to_numpy()[:, :2]
    pm_pos[:, 1] += offsets['p']
    
    # Generate negative pairs
    neg_ratings = pm[pm['rating'] == 0].to_numpy()[:, :2]
    neg_ratings[:, 1] += offsets['p']
    assert (pm_pos.shape[0] + neg_ratings.shape[0] == pm.shape[0])
    
    # Shuffle and select negative pairs
    indices_neg = np.arange(neg_ratings.shape[0])
    np.random.shuffle(indices_neg)
    indices_neg = indices_neg[:pm_pos.shape[0] * 1]
    neg_data = neg_ratings[indices_neg]
    np.savez("../preprocessed/neg_pairs_offset", neg_data=neg_data)
    
    # Shuffle positive pairs
    indices = np.arange(pm_pos.shape[0])
    np.random.shuffle(indices)
    pos_data = pm_pos[indices]
    np.savez("../preprocessed/pos_pairs_offset", pos_data=pos_data)
    print('==========Step 3 complete==========')

    adjs_offset = {}
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    adj_offset[pm_pos[:, 0], pm_pos[:, 1]] = 1
    adjs_offset['pm'] = sp.coo_matrix(adj_offset)
    # Create adjacency matrix for peptide-peptide relation
    pp_npy = pp.to_numpy(int)[:, :2]
    pp_matrix = np.zeros((max_pid, max_pid), dtype=float)
    pp_score = pp['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(pp_npy[:, 0], pp_npy[:, 1], pp_score):
        adj_offset[i, j] = k
        pp_matrix[i, j] = k
        adj_offset[j, i] = k
        pp_matrix[j, i] = k
    adjs_offset['simpp'] = sp.coo_matrix(adj_offset)
    # Create adjacency matrix for microbe-microbe relation
    mm_npy = mm.to_numpy(int)[:, :2]
    mm_matrix = np.zeros((max_mid, max_mid), dtype=float)
    mm_score = mm['weight'].tolist()
    adj_offset = np.zeros((node_types.shape[0], node_types.shape[0]), dtype=np.float32)
    for i, j, k in zip(mm_npy[:, 0] + offsets['p'], mm_npy[:, 1] + offsets['p'], mm_score):
        adj_offset[i, j] = k
        mm_matrix[i - offsets['p'], j - offsets['p']] = k
    adjs_offset['simmm'] = sp.coo_matrix(adj_offset)
    print('==========Step 4 complete==========')
    # Save combined matrices
    np.savez('../preprocessed/combined_matrices.npz', dp_matrix=pp_matrix, pd_matrix=mm_matrix)
    
    # Save adjacency matrices
    with open("../preprocessed/adjs_offset.pkl", "wb") as f2:
        pickle.dump(adjs_offset, f2)
    print('==========Step 5 complete==========')
if __name__ == '__main__':
    prefix = "./data/PMDHAN"
    pretreatment_Pep(prefix)
    # pretreatment_drugmicrobe(prefix)
