import os
import statistics

import numpy as np
import pickle
import scipy.sparse as sp
import argparse
import torch
import torch.nn.functional as F
import torch.nn as nn
from sklearn.metrics import roc_auc_score, average_precision_score, accuracy_score, f1_score, precision_score, \
    recall_score

from model import Model
from preprocess import normalize_sym, normalize_row, sparse_mx_to_torch_sparse_tensor
import train_search
from dataset import get_datafold
from RWR import generate_node_embeddings
os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 下面老是报错 shape 不一致

parser = argparse.ArgumentParser()
parser.add_argument('--lr', type=float, default=0.006, help='learning rate')
parser.add_argument('--wd', type=float, default=0.06, help='weight decay')
parser.add_argument('--n_hid', type=int, default=64, help='hidden dimension')
parser.add_argument('--dataset', type=str, default='Drug')
parser.add_argument('--gpu', type=int, default=0)
parser.add_argument('--epochs', type=int, default=110, help='number of training epochs')
parser.add_argument('--dropout', type=float, default=0.2)
parser.add_argument('--seed', type=int, default=1)
args = parser.parse_args()

prefix = "lr" + str(args.lr) + "_wd" + str(args.wd) + "_h" + str(args.n_hid) + \
         "_drop" + str(args.dropout) + "_epoch" + str(args.epochs) + "_cuda" + str(args.gpu)

archs = {
    # 未删除
    "source": ([[4, 4, 0]], [[1, 8, 8]]),
    "target": ([[1, 4, 1]], [[3, 0, 8]])
}


def main():
    torch.cuda.set_device(args.gpu)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)


    # archs["source"], archs["target"] = train_search.search_main()
    steps_s = [len(meta) for meta in archs["source"][0]]
    steps_t = [len(meta) for meta in archs["target"][0]]

    prefix = "./preprocessed/"

    #* load data
    node_types = np.load(os.path.join(prefix, "node_types.npy"))
    num_node_types = node_types.max() + 1
    node_types = torch.from_numpy(node_types).cuda()

    adjs_offset = pickle.load(open(os.path.join(prefix, "adjs_offset.pkl"), "rb"))
    adj_key = list(adjs_offset.keys())
    adjs_pt = []
    for k in adj_key:
        if 'sim' in k:
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                normalize_sym(adjs_offset[k] + sp.eye(adjs_offset[k].shape[0], dtype=np.float32))).cuda())
        else:
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                normalize_row(adjs_offset[k] + sp.eye(adjs_offset[k].shape[0], dtype=np.float32))).cuda())
            adjs_pt.append(sparse_mx_to_torch_sparse_tensor(
                normalize_row(adjs_offset[k].T + sp.eye(adjs_offset[k].shape[0], dtype=np.float32))).cuda())
    adjs_pt.append(sparse_mx_to_torch_sparse_tensor(sp.eye(adjs_offset[adj_key[0]].shape[0], dtype=np.float32).tocoo()).cuda())
    adjs_pt.append(torch.sparse.FloatTensor(size=adjs_offset[adj_key[0]].shape).cuda())
    print("Loading {} adjs...".format(len(adjs_pt)))

    #* load labels
    pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold = get_datafold()

    # * embedding
    in_dims = []
    # node_feats = []
    node_feats = torch.load(os.path.join(prefix, "node_feats.pt"))
    print("Load Over!")

    model_s = Model(in_dims, args.n_hid, steps_s, dropout=args.dropout).cuda()
    model_t = Model(in_dims, args.n_hid, steps_t, dropout=args.dropout).cuda()

    optimizer = torch.optim.Adam(
        list(model_s.parameters()) + list(model_t.parameters()),
        lr=args.lr,
        weight_decay=args.wd
    )

    auc_best = None
    aupr_best = None
    auc_mean = []
    auprc_mean = []
    for pos_train, pos_val, pos_test, neg_train, neg_val, neg_test in zip(pos_train_fold, pos_val_fold, pos_test_fold,
                                                                          neg_train_fold, neg_val_fold, neg_test_fold):

        for epoch in range(args.epochs):
            train_loss = train(node_feats, node_types, adjs_pt, pos_train, neg_train, model_s, model_t, optimizer)
            auc_test, aupr = infer(node_feats, node_types, adjs_pt, pos_val, neg_val, pos_test, neg_test, model_s, model_t)
            # print("第{}轮; loss：{}; auc：{}".format(epoch + 1, train_loss, auc_test))
            if auc_best is None or auc_test > auc_best:
                auc_best = auc_test
                aupr_best = aupr
        auc_mean.append(auc_best)
        auprc_mean.append(aupr_best)
        print("AUC：{:.3f}".format(auc_best))
        print("AUPR：{:.3f}".format(aupr_best))
        # break
    print("AUC_mean：{:.3f}".format(statistics.mean(auc_mean)))
    print("AUPR_mean：{:.3f}".format(statistics.mean(auprc_mean)))

def train(node_feats, node_types, adjs, pos_train, neg_train, model_s, model_t, optimizer):
    model_s.train()
    model_t.train()
    optimizer.zero_grad()

    out_s = model_s(node_feats, node_types, adjs, archs["source"][0], archs["source"][1])
    out_t = model_t(node_feats, node_types, adjs, archs["target"][0], archs["target"][1])

    loss = - torch.mean(F.logsigmoid(torch.mul(out_s[pos_train[:, 0]], out_t[pos_train[:, 1]]).sum(dim=-1)) + \
                        F.logsigmoid(- torch.mul(out_s[neg_train[:, 0]], out_t[neg_train[:, 1]]).sum(dim=-1)))
    loss.backward()
    optimizer.step()
    return loss.item()

def infer(node_feats, node_types, adjs, pos_val, neg_val, pos_test, neg_test, model_s, model_t):
    model_s.eval()
    model_t.eval()
    with torch.no_grad():
        out_s = model_s(node_feats, node_types, adjs, archs["source"][0], archs["source"][1])
        out_t = model_t(node_feats, node_types, adjs, archs["target"][0], archs["target"][1])

    pos_test_prod = torch.mul(out_s[pos_test[:, 0]], out_t[pos_test[:, 1]]).sum(dim=-1)
    neg_test_prod = torch.mul(out_s[neg_test[:, 0]], out_t[neg_test[:, 1]]).sum(dim=-1)

    y_true_test = np.zeros((pos_test.shape[0] + neg_test.shape[0]), dtype=np.int64)
    y_true_test[:pos_test.shape[0]] = 1
    y_pred_test = np.concatenate(
        (torch.sigmoid(pos_test_prod).cpu().numpy(), torch.sigmoid(neg_test_prod).cpu().numpy()))
    auc = roc_auc_score(y_true_test, y_pred_test)
    aupr = average_precision_score(y_true_test, y_pred_test)
    return auc, aupr

if __name__ == '__main__':
    main()
