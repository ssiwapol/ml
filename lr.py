import sys
import csv

import numpy as np


class LogisticRegression:
    '''
    Logistic Regression model
    Binary classification
    
    Parameters:
    x (array): feature array, size=(n_obs, n_feat)
    y (array): label array, size=(n_obs)
    '''
    def __init__(self, x, y):
        self.x = x
        self.y = y
    
    # calculate sigmoid of x
    def sigmoid(self, x):
        return 1/(1 + np.exp(-x))

    # caculate phi (sigmoid of w.T dot x)
    def phi(self, w, x):
        return self.sigmoid(np.dot(w.T, x))

    # calulate sgd at i-th data points
    def sgd(self, x, y, w, i):
        n = x.shape[0]
        return (-1/n) * (y[i] - self.phi(w, x[i])) * x[i]
    
    # train model 1 epoch
    def fit1(self, w, lr=0.01):
        for i in range(self.x.shape[0]):
            g = self.sgd(self.x, self.y, w, i)
            w = w - (lr * g)
        self.w = w
        return w

    # train model, by updating weight
    def fit(self, epoch, lr=0.01, verbose=False):
        w = np.zeros(self.x.shape[1])
        for e in range(epoch):
            if verbose:
                print("train: {}/{}".format(e+1, epoch))
            for i in range(self.x.shape[0]):
                g = self.sgd(self.x, self.y, w, i)
                w = w - (lr * g)
        self.w = w
    
    # predict x array
    def pred(self, x, threshold=0.5):
        y_pred = [self.phi(self.w, i) for i in x]
        y_pred = [1 if i >= threshold else 0 for i in y_pred]
        return np.array(y_pred)
    
    # predict and report error
    def pred_error(self, x, y, threshold=0.5):
        y_pred = self.pred(x, threshold)
        y_true = y
        err = np.mean((y_true != y_pred))
        return y_pred, err


############# UTIL FUNC #############
# read data from file
def read_file(filepath):
    # read file into numpy array
    with open(filepath, 'r') as f:
        data = list(csv.reader(f, delimiter="\t"))
    arr = np.array([r for r in data])
    arr = arr.astype(float)
    # trainsform input into x, y and add intercept
    y = arr[:,0]
    x = arr[:,1:]
    x0 = np.reshape(np.ones(x.shape[0]), (-1,1))
    x = np.append(x0, x, axis=1)
    return x, y

# write list into file
def write_list(list_val, filepath):
    with open(filepath, 'w') as f:
        for x in list_val:
            f.write(str(x) + "\n")

# write metrics file
def write_evl(err_trn, err_tst, filepath):
    with open(filepath, 'w') as f:
        f.write('error(train): {:.6f}\n'.format(err_trn))
        f.write('error(test): {:.6f}'.format(err_tst))


# python3 lr.py train.tsv valid.tsv test.tsv dict.txt train.labels test.labels metrics.txt 500
if __name__=='__main__':
    '''
    Read/write data in tsv file format (without column)
    x11 x12 x13 y1
    x21 x22 x23 y2
    '''
    # cli arguments
    formatted_train_input = sys.argv[1]
    formatted_validation_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])
    # run model
    x_trn, y_trn = read_file(formatted_train_input)
    lr = LogisticRegression(x_trn, y_trn)
    lr.fit(epoch=num_epoch)
    # predict train
    y_pred, err_trn = lr.pred_error(lr.x, lr.y)
    write_list(list(y_pred), train_out)
    # predict test
    x_tst, y_tst = read_file(formatted_test_input)
    y_pred, err_tst = lr.pred_error(x_tst, y_tst)
    write_list(list(y_pred), test_out)
    # write metrics
    write_evl(err_trn, err_tst, metrics_out)
