import sys
import csv

import numpy as np


class NeuralNet():
    '''
    Deep Learning model
    Multi-class classification

    Parameters:
    x (array): feature array, size=(n_obs, n_feat)
    y (array): label array, size=(n_obs)

    Structure as followed:
        linear feed forward
        sigmoid activation function
        linear feed forward
        softmax activation function
        cross-entropy loss function
        update weight by adagrad
    '''
    def __init__(self, x, y, hidden_units, init_flag=2):
        self.x = self.add_bias(x)
        self.y = y.astype(int)
        self.y = np.eye(max(self.y) + 1)[self.y]
        self.l1 = Linear(self.x.shape[1], hidden_units, init_flag, False)
        self.a1 = Sigmoid(self.x.shape[1])
        self.l2 = Linear(hidden_units+1, self.y.shape[1], init_flag, True)
        self.a2 = Softmax(self.y.shape[1])
        self.j = CrossEntropy()

    # add bias term to the data
    def add_bias(self, x):
        if x.ndim <= 1:
            o = np.ones(1)
        else:
            o = np.ones(x.shape[0]).reshape(-1,1)
        return np.hstack((o, x))

    # train on i row
    def train(self, i, lr):
        # select row i
        x = self.x[i].reshape(1, -1)
        y = self.y[i].reshape(1, -1)
        # forward computing
        a = self.l1.forward(x)
        z = self.a1.forward(a)
        z = self.add_bias(z)
        b = self.l2.forward(z)
        yh = self.a2.forward(b)
        # self.loss = self.j.forward(yh, y)
        # backward computing
        self.a2.backward(yh, y)
        self.l2.backward(self.a2.dl, z)
        self.a1.backward(self.l2.dl, z[:, 1:])
        self.l1.backward(self.a1.dl, x)
        # update weight
        self.l1.update_w(lr)
        self.l2.update_w(lr)
    
    # batch forward prediction
    def pred(self, x, y):
        x1 = self.add_bias(x)
        y1 = y.astype(int)
        y1 = np.eye(self.y.shape[1])[y1]
        a = self.l1.forward(x1)
        z = self.a1.forward(a)
        z = self.add_bias(z)
        b = self.l2.forward(z)
        yh = self.a2.forward(b)
        loss = self.j.forward(yh, y1)
        y_pred = np.argmax(yh, axis=1)
        y_pred = y_pred.astype(float)
        err = np.mean(np.not_equal(y, y_pred))
        return y_pred, loss, err

def run(x_trn, y_trn, x_val, y_val, hidden_units, lr, n_epoch, init_flag=2, metric_file=None, verbose=False):
    # remove existing metric file
    if metric_file:
        with open(metric_file, 'w') as fp:
            pass
    # init NeuralNet class
    model = NeuralNet(x_trn, y_trn, hidden_units, init_flag)
    # train model using sgd method
    for i in range(n_epoch):
        for j in range(model.x.shape[0]):
            model.train(j, lr)
        # write and display result during training
        y_pred_trn, loss_trn, err_trn = model.pred(x_trn, y_trn)
        y_pred_val, loss_val, err_val = model.pred(x_val, y_val)
        loss_txt_trn = "epoch={} crossentropy (train): {}".format(i+1, loss_trn)
        loss_txt_val = "epoch={} crossentropy (validation): {}".format(i+1, loss_val)
        if metric_file:
            append_file(loss_txt_trn, metric_file)
            append_file(loss_txt_val, metric_file)
        if verbose:
            print(loss_txt_trn)
            print(loss_txt_val)
    err_txt_trn = "error (train): {}".format(err_trn)
    err_txt_val = "error (validation): {}".format(err_val)
    if metric_file:
        append_file(err_txt_trn, metric_file)
        append_file(err_txt_val, metric_file)
    if verbose:
        print(err_txt_trn)
        print(err_txt_val)
    return y_pred_trn, y_pred_val


############# NN FUNCTION #############
class Linear():
    def __init__(self, input_dim, output_dim, init_flag=2, hidden=True):
        if init_flag==1:
            self.w = np.random.uniform(-0.1, 0.1, (output_dim, input_dim))
        else:
            self.w = np.zeros((output_dim, input_dim))
        self.dw = np.zeros((output_dim, input_dim))
        self.dl = np.zeros((1, input_dim))
        self.hidden = hidden
        self.st = np.zeros((output_dim, input_dim))

    def forward(self, x):
        return np.matmul(x, self.w.T)

    def backward(self, dl, x):
        self.dw = np.matmul(dl.T, x)
        if self.hidden:
            self.dl = np.matmul(dl, self.w[:, 1:])
        else:
            self.dl = np.matmul(dl, self.w)

    def update_w(self, lr):
        epsilon = 1e-5
        self.st = self.st + np.square(self.dw)
        self.w = self.w - np.multiply(np.divide(lr, np.sqrt(self.st + epsilon)), self.dw)

class Sigmoid():
    def __init__(self, input_dim):
        self.dl = np.zeros((1, input_dim))

    def forward(self, x):
        return 1/(1 + np.exp(-x))

    def backward(self, dl, x):
        self.dl = np.multiply(dl, np.multiply(x, (1-x)))

class Softmax():
    def __init__(self, output_dim):
        self.dl = np.zeros((1, output_dim))

    def forward(self, x):
        return np.exp(x) / np.sum(np.exp(x), axis=1).reshape(-1, 1)

    def backward(self, yh, y):
        self.dl = (yh - y).reshape(1, -1)

class CrossEntropy():
    def __init__(self):
        pass

    def forward(self, yh, y):
        l = np.sum(np.multiply(-y, np.log(yh)), axis=1)
        return np.sum(l) / l.size


############# UTIL FUNC #############
def read_file(filepath):
    # read file into numpy array
    with open(filepath, 'r') as f:
        data = list(csv.reader(f, delimiter=","))
    arr = np.array([r for r in data])
    y_dict = {float(i): i for i in np.unique(arr[:, 0])}
    arr = arr.astype(float)
    y = arr[:,0]
    x = arr[:,1:]
    return x, y, y_dict

def append_file(txt, filepath):
    with open(filepath, 'a') as f:
        f.write(txt + "\n")

def write_list(list_val, filepath):
    with open(filepath, 'w') as f:
        for x in list_val:
            f.write(str(x) + "\n")


# python3 nn.py train.csv val.csv train.labels val.labels metrics.txt 2 4 2 0.1
if __name__=='__main__':
    '''
    Read/write data in csv file format (without column)
    x11,x12,x13,y1
    x21,x22,x23,y2
    '''
    # cli arguments
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    train_out = sys.argv[3]
    validation_out = sys.argv[4]
    metrics_out = sys.argv[5]
    num_epoch = int(sys.argv[6])
    hidden_units = int(sys.argv[7])
    init_flag = int(sys.argv[8])
    learning_rate = float(sys.argv[9])
    # run model
    x_trn, y_trn, y_dict = read_file(train_input)
    x_val, y_val, _ = read_file(validation_input)
    y_pred_trn, y_pred_val = run(x_trn, y_trn, x_val, y_val, hidden_units, learning_rate, num_epoch, init_flag, metrics_out, False)
    y_pred_trn = [y_dict[i] for i in y_pred_trn]
    y_pred_val = [y_dict[i] for i in y_pred_val]
    write_list(y_pred_trn, train_out)
    write_list(y_pred_val, validation_out)
