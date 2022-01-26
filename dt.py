import sys

import numpy as np


class DecisionTree:
    '''
    Deicion Tree algorithm
    Multi-class classification
    Features are categorical data
    
    Parameters:
    col (list): 1-d column name, size=(n_feat+n_label)
    data (list): 2-d data, size=(n_obs, n_feat+n_label)
    '''
    def __init__(self, col, data):
        # feature data
        self.feat = {}
        for i in range(len(col[:-1])):
            self.feat[i] = {}
            self.feat[i]['i'] = i
            self.feat[i]['name'] = col[i]
            feat_key = sorted(list(set(j[i] for j in data)))
            self.feat[i]['key'] = {k: v for k, v in zip(feat_key, range(len(feat_key)))}
            self.feat[i]['val'] = {v: k for k, v in self.feat[i]['key'].items()}
        # label data
        self.label = {}
        self.label['name'] = col[-1]
        label_key = sorted(list(set(i[-1] for i in data)))
        self.label['key'] = {k: v for k, v in zip(label_key, range(len(label_key)))}
        self.label['val'] = {v: k for k, v in self.label['key'].items()}
        # transform data to array
        self.x_trn = np.array([[self.feat[j]['key'].get(i[j]) for j in self.feat] for i in data])
        self.y_trn = np.array([self.label['key'].get(i[-1]) for i in data])

    # caluculate entropy of array
    def cal_entropy(self, y):
        _, c = np.unique(y, return_counts=True)
        h = c / y.size
        h = np.multiply(-h, np.log2(h))
        return h.sum()

    # calculate mutual information of x on y
    def cal_mutualinfo(self, x, y):
        hy = self.cal_entropy(y)
        u, c = np.unique(x, return_counts=True)
        p = c / x.size
        hx = [self.cal_entropy(y[x==j]) for j in u]
        hx = np.multiply(hx, p).sum()
        return hy - hx

    # find the node that has maximum mutual information
    def max_mutualinfo(self, x, y, feat):
        mi = {i: self.cal_mutualinfo(x[:, i], y) for i in feat}
        # when the mutual information are all equal
        if len(set([i for i in mi.values()])) <= 1:
            i_max = min(mi)
        else:
            i_max = max(mi, key=mi.get)
        return i_max, mi[i_max]

    # calculate majority vote
    def cal_majority(self, y):
        u, c = np.unique(y, return_counts=True)
        if len(u) <= 0:
            return None
        # if the vote is tied, return last in the lexicographical order
        if len(set(c)) <= 1:
            return max(u)
        v = {i: j for i, j in zip(u, c)}
        return max(v, key=v.get)

    # find label
    def get_stats(self, y, label_val):
        u, c = np.unique(y, return_counts=True)
        v = {k: v for k, v in zip(u, c)}
        label = ['{} {}'.format(v.get(i, 0), label_val[i]) for i in label_val]
        label = '/'.join(label)
        return label

    # train model
    def train(self, x, y, feat, label, max_depth, depth=0, node=0):
        node_i, mi = self.max_mutualinfo(x, y, feat)
        stats = self.get_stats(y, label)
        if mi <= 0:
            return Leaf(self.cal_majority(y), feat[node], stats)
        if max_depth >= 0:
            if depth >= max_depth:
                return Leaf(self.cal_majority(y), feat[node], stats)
        depth = depth + 1
        branch = {}
        for i in feat[node_i]['val']:
            branch[i] = self.train(x[x[:, node_i]==i], y[x[:, node_i]==i], feat, label, max_depth, depth, node_i)
        return Node(node_i, feat[node_i], branch, depth, stats)
    
    # fit model
    def fit(self, max_depth=-1, verbose=True):
        self.model = self.train(self.x_trn, self.y_trn, self.feat, self.label['val'], max_depth)
        if verbose:
            self.model.printTree()

    # predict one data
    def pred_one(self, x, model):
        if isinstance(model, Leaf):
            return model.y
        else:
            #print('feat: {} | val: {}'.format(feat_i, x_val))
            feat_i = model.feat_i
            x_val = x[feat_i]
            model = model.branch[x_val]
            return self.pred_one(x, model)
    
    # predict from array
    def pred_arr(self, x):
        return [self.pred_one(i, self.model) for i in x]

    # predict and calculate error
    def pred(self, data):
        x, y = self.trans(data)
        y_pred = self.pred_arr(x)
        y_true = y
        err = np.mean((y_true != y_pred))
        y_pred_label = [self.label['val'].get(i) for i in y_pred]
        return y_pred_label, err

    # transform data into array
    def trans(self, data):
        # transform data to file
        x = np.array([[self.feat[j]['key'].get(i[j]) for j in self.feat] for i in data])
        y = np.array([self.label['key'].get(i[-1]) for i in data])
        return x, y


class Node:
    def __init__(self, feat_i, attr, branch, depth, stats):
        self.feat_i = feat_i
        self.attr = attr
        self.branch = branch
        self.depth = depth
        self.stats = stats

    def __str__(self):
        return '{} ({})\ndepth: {}\ntotal branch: {}'.format(
            self.feat_i,
            self.attr['name'],
            self.depth,
            len(self.branch)
        )

    def printBranch(self):
        for i in self.branch:
            print('{}{} = {}: [{}]'.format(
                '| ' * self.depth, 
                self.attr['name'], 
                self.branch[i].attr['val'].get(i), 
                self.branch[i].stats)
                )
            if isinstance(self.branch[i], Node):
                self.branch[i].printBranch()

    def printTree(self):
        print('[{}]'.format(self.stats))
        self.printBranch()


class Leaf:
    def __init__(self, y, attr, stats):
        self.y = y
        self.attr = attr
        self.stats = stats

    def __str__(self):
        return 'y={}\nnode: {} ({})'.format(
            self.y,
            self.attr['i'],
            self.attr['name']
        )
    
    def printTree(self):
        print('[{}]'.format(self.stats))


############# UTIL FUNC #############
# read file to list
def read_file(filepath):
    with open(filepath, 'r') as f:
        c = f.readlines()
    col = [x.split() for x in c][0]
    data = [x.split() for x in c][1:]
    return col, data

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


# python3 dt.py train.tsv test.tsv 3 train_out.tsv test_out.tsv metrics.txt
if __name__ == '__main__':
    '''
    read/write data in tsv file format
    feat1 feat2 feat3 label
    x11 x12 x13 y1
    x21 x22 x23 y2
    '''
    # cli arguments
    train_input = sys.argv[1]
    test_input = sys.argv[2]
    max_depth = int(sys.argv[3])
    train_out = sys.argv[4]
    test_out = sys.argv[5]
    metrics_out = sys.argv[6]
    # run model
    col, trn_data = read_file(train_input)
    dt = DecisionTree(col, trn_data)
    dt.fit(max_depth, True)
    y_pred, trn_err = dt.pred(trn_data)
    write_file(y_pred, train_out)
    _, tst_data = read_file(test_input)
    y_pred, tst_err = dt.pred(tst_data)
    write_file(y_pred, test_out)
    write_evl(metrics_out, trn_err, tst_err)
