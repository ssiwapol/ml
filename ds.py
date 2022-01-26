import sys

import numpy as np


class DecisionStump:
    '''
    Deicion Stump algorithm
    Multi-class classification
    Features are categorical data

    Parameters:
    x (array): feature array, size=(n_obs, n_feat)
    y (array): label array, size=(n_obs)
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y

    # fit model
    def fit(self, split_index):
        # calculate majority vote of feature at split_index
        self.split_index = split_index
        arr = np.concatenate((self.x, self.y.reshape(-1, 1)), axis=1)
        self.model = {}
        for i in np.unique(arr[:, self.split_index]):
            m = 0
            for j in np.unique(arr[:,-1]):
                a = arr[np.where(arr[:,self.split_index]==i)]
                cnt = len(a[np.where(a[:,-1]==j)])
                if cnt >= m:
                    m = cnt
                    self.model[i] = j

    # predict
    def pred(self, x, y):
        # prepare x
        x = x[:, self.split_index]
        # predict y
        y_pred = np.array([self.model.get(i) for i in x])
        err = np.mean((y != y_pred))
        return y_pred, err


############# UTIL FUNC #############
def read_file(filepath):
    # read file
    with open(filepath, 'r') as f:
        data = f.readlines()[1:]
    data = [x.split() for x in data]
    # prep data
    feat = {}
    for i in range(len(data[0][:-1])) :
        feat[i] = {}
        feat_key = sorted(list(set(j[i] for j in data)))
        feat[i]['key'] = {k: v for k, v in zip(feat_key, range(len(feat_key)))}
        feat[i]['val'] = {v: k for k, v in feat[i]['key'].items()}
    label = {}
    label_key = sorted(list(set(i[-1] for i in data)))
    label['key'] = {k: v for k, v in zip(label_key, range(len(label_key)))}
    label['val'] = {v: k for k, v in label['key'].items()}
    x = np.array([[feat[j]['key'].get(i[j]) for j in feat] for i in data])
    y = np.array([label['key'].get(i[-1]) for i in data])
    return x, y, feat, label

# write list to file
def write_file(list_val, filepath):
    with open(filepath, 'w') as f:
        for x in list_val:
            f.write(x + "\n")

# write evaluated result
def write_evl(filepath, trn_err, tst_err):
    with open(filepath, 'w') as f:
        f.write(f'error(train): {trn_err}\n')
        f.write(f'error(test): {tst_err}')


# python3 ds.py train.tsv test.tsv 3 train.labels test.labels 3_metrics.txt
if __name__ == '__main__':
    '''
    Read/write data in tsv file format
    feat1 feat2 feat3 label
    x11 x12 x13 y1
    x21 x22 x23 y2
    '''
    # cli arguments
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    split_index = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    # run model
    x_trn, y_trn, feat, label = read_file(train_input)
    ds = DecisionStump(x_trn, y_trn)
    ds.fit(split_index)
    y_pred, trn_err = ds.pred(x_trn, y_trn)
    write_file([label['val'].get(i) for i in y_pred], train_out)
    x_tst, y_tst, _, _ = read_file(test_input)
    y_pred, tst_err = ds.pred(x_trn, y_trn)
    write_file([label['val'].get(i) for i in y_pred], test_out)
    write_evl(metrics_out, trn_err, tst_err)
