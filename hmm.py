import sys

import numpy as np


class ForwardBackward:
    '''
    HMM Forward-Backward prediction
    input 
        initial probability matrix, 
        emission matrix, 
        transition matrix
    option to calculate in log-space
    '''
    def __init__(self, pi, a, b):
        self.pi = pi
        self.a = a
        self.b = b
        self.pi_l = np.log(pi)
        self.a_l = np.log(a)
        self.b_l = np.log(b)

    # log-sum-trick to avoid underflow
    def log_sum_exp(self, arr):
        m = np.max(arr)
        return m + np.log(np.sum(np.exp(arr - m), axis=1))

    # forward calculation without log-space
    def forward(self, x, j):
        aj = self.a[:, x[j]]
        if j==0:
            return self.pi * aj
        else:
            return aj * np.matmul(self.b.T, self.forward(x, j-1))

    # forward calculation with log-space
    def forward_l(self, x, j):
        aj = self.a_l[:, x[j]]
        if j==0:
            return self.pi_l + aj
        else:
            return aj + self.log_sum_exp(self.b_l.T + self.forward_l(x, j-1))

    # backward calculation without log-space
    def backward(self, x, j):
        if j==len(x)-1:
            return np.ones(self.a.shape[0])
        else:
            aj = self.a[:, x[j+1]]
            return np.matmul(self.b, aj * self.backward(x, j+1))

    # backward calculation with log-space
    def backward_l(self, x, j):
        if j==len(x)-1:
            return np.zeros(self.a_l.shape[0])
        else:
            aj = self.a_l[:, x[j+1]]
            return self.log_sum_exp(self.b_l + aj + self.backward_l(x, j+1))

    # predict from array of index
    def pred(self, x, log_space=True):
        alpha = np.zeros((len(x), self.pi.shape[0]))
        beta = np.zeros((len(x), self.pi.shape[0]))
        if log_space:
            for j in range(len(x)):
                alpha[j] = self.forward_l(x, j)
                beta[j] = self.backward_l(x, j)
            y_hat = alpha + beta
            y_pred = np.argmax(y_hat, axis=1)
            ll = self.log_sum_exp(alpha[-1].reshape(1, -1))[0]
        else:
            for j in range(len(x)):
                alpha[j] = self.forward(x, j)
                beta[j] = self.backward(x, j)
            px = np.sum(alpha[-1])
            y_hat = alpha * beta / px
            y_pred = np.argmax(y_hat, axis=1)
            ll = np.array([np.log(np.sum(alpha[-1]))])[0]
        return y_pred, ll

# read file
def read_file(validation_input, index_to_word, index_to_tag, hmminit, hmmemit, hmmtrans):
    with open(validation_input, 'r') as f:
        val_data = []
        a = []
        for i in f:
            if i != '\n':
                a.append(tuple(i.split()))
            else:
                val_data.append(a)
                a = []
        # append last data (no '\n')
        val_data.append(a)
    with open(index_to_word, 'r') as f:
        word_list = [i.rstrip() for i in f]
    with open(index_to_tag, 'r') as f:
        tag_list = [i.rstrip() for i in f]
    with open(hmminit, 'r') as f:
        pi = np.array(list(f)).reshape(-1,)
        pi = pi.astype(float)
    with open(hmmemit, 'r') as f:
        a = np.array([i.split() for i in list(f)])
        a = a.astype(float)
    with open(hmmtrans, 'r') as f:
        b = np.array([i.split() for i in list(f)])
        b = b.astype(float)
    return val_data, word_list, tag_list, pi, a, b

# run prediction
def run(val_data, word_list, tag_list, pi, a, b, verbose=False):
    m = ForwardBackward(pi, a, b)
    y_pred_list = []
    ll_list = []
    n = 0
    e = 0
    for i in range(len(val_data)):
        if verbose:
            print("{}/{}".format(i, len(val_data)))
        x = np.array([word_list.index(i[0]) for i in val_data[i]])
        y_pred, ll = m.pred(x, log_space=True)
        y_true = [tag_list.index(j[1]) for j in val_data[i]]
        e += np.sum(np.array(y_pred) != np.array(y_true))
        ll_list.append(ll)
        n += len(val_data[i])
        y_pred_label = [tag_list[i] for i in y_pred]
        y_pred_list.append([(i[0], j) for i, j in zip(val_data[i], y_pred_label)])
    avg_ll = np.mean(ll_list)
    acc = 1 - (e / n)
    return y_pred_list, avg_ll, acc

# write output
def write_file(predicted_file, metric_file, y_pred_list, avg_ll, acc):
    with open(predicted_file, 'w') as f:
        for i in range(len(y_pred_list)):
            for j in range(len(y_pred_list[i])):
                f.write('{}\t{}'.format(y_pred_list[i][j][0], y_pred_list[i][j][1]) + '\n')
            f.write('\n')
    with open(metric_file, 'w') as f:
        f.write('Average Log-Likelihood: {}'.format(avg_ll)+ '\n')
        f.write('Accuracy: {}'.format(acc) + '\n')


# python3 forwardbackward.py val.txt index_to_word.txt index_to_tag.txt hmminit.txt hmmemit.txt hmmtrans.txt predicted.txt metrics.txt
if __name__=='__main__':
    '''
    Read/write data from matrix file
    '''
    # cli arguments
    validation_input = sys.argv[1]
    index_to_word = sys.argv[2]
    index_to_tag = sys.argv[3]
    hmminit = sys.argv[4]
    hmmemit = sys.argv[5]
    hmmtrans = sys.argv[6]
    predicted_file = sys.argv[7]
    metric_file = sys.argv[8]
    # run model
    val_data, word_list, tag_list, pi, a, b = read_file(validation_input, index_to_word, index_to_tag, hmminit, hmmemit, hmmtrans)
    y_pred_list, avg_ll, acc = run(val_data, word_list, tag_list, pi, a, b)
    write_file(predicted_file, metric_file, y_pred_list, avg_ll, acc)
