import os

import numpy as np
from sklearn.model_selection import KFold

def get_datafold(): # seed：3
    kf = KFold(n_splits=5, shuffle=True, random_state=2)
    pos = np.load("./preprocessed/pos_pairs_offset.npz")
    pos_data = pos['pos_data']

    neg = np.load("./preprocessed/neg_pairs_offset.npz")
    neg_data = neg['neg_data']
    pos_train_fold = []
    pos_val_fold = []
    pos_test_fold = []
    neg_train_fold = []
    neg_val_fold = []
    neg_test_fold = []
    for (pos_train_index, pos_test_index), (neg_train_index, neg_test_index) in zip(kf.split(pos_data),
                                                                                    kf.split(neg_data)):
        # 划分训练集和测试集
        pos_train_data = pos_data[pos_train_index]  # 训练集
        pos_test_data = pos_data[pos_test_index]  # 测试集
        neg_train_data = neg_data[neg_train_index]
        neg_test_data = neg_data[neg_test_index]

        train_len = int(len(pos_train_data) * 0.6)
        pos_train_data, pos_val_data = pos_train_data[:train_len], pos_train_data[train_len:]
        pos_train_fold.append(pos_train_data)
        pos_val_fold.append(pos_val_data)
        pos_test_fold.append(pos_test_data)

        neg_train_data, neg_val_data = neg_train_data[:train_len], neg_train_data[train_len:]
        neg_train_fold.append(neg_train_data)
        neg_val_fold.append(neg_val_data)
        neg_test_fold.append(neg_test_data)

    return pos_train_fold, pos_val_fold, pos_test_fold, neg_train_fold, neg_val_fold, neg_test_fold