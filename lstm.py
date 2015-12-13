import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters
import scipy.linalg

def transition_init(size,fan_out):
    W = np.empty((size,fan_out * size),dtype=np.float32)
    for i in xrange(fan_out):
        W[:,i * size:(i + 1) * size] = scipy.linalg.orth(np.random.randn(size,size))
    return W



def build(P, name, input_size, hidden_size, truncate_gradient=-1):
    name_init_hidden = "init_%s_hidden" % name
    name_init_cell = "init_%s_cell" % name
    P[name_init_hidden] = np.zeros((hidden_size,))
    P[name_init_cell] = np.zeros((hidden_size,))

    _step = build_step(P, name, input_size, hidden_size)

    def lstm_layer(X):
        init_hidden = T.tanh(P[name_init_hidden])
        init_cell = P[name_init_cell]
        init_hidden_batch = T.alloc(init_hidden, X.shape[1], hidden_size)
        init_cell_batch = T.alloc(init_cell, X.shape[1], hidden_size)
        [cell, hidden], _ = theano.scan(
            _step,
            sequences=[X],
            outputs_info=[init_cell_batch, init_hidden_batch],
            truncate_gradient=truncate_gradient
        )
        return cell, hidden
    return lstm_layer


def build_step(P, name, input_sizes, hidden_size):
    name_W_hidden = "W_%s_hidden" % name
    name_W_cell = "W_%s_cell" % name
    name_b = "b_%s" % name

    W_inputs = []
    for i,input_size in enumerate(input_sizes):
        name_W = "W_%s_input_%d"%(name,i)
        P[name_W] = 0.08 * np.random.rand(input_size, hidden_size * 4)
        W_inputs.append(P[name_W])

    P[name_W_hidden] = transition_init(hidden_size,4)
    P[name_W_cell] = transition_init(hidden_size,3)
    bias_init = np.zeros((4, hidden_size), dtype=np.float32)
    bias_init[1] = 0
    P[name_b] = bias_init

    V_if = P[name_W_cell][:, 0 * hidden_size:2 * hidden_size]
    V_o = P[name_W_cell][:, 2 * hidden_size:3 * hidden_size]

    biases = P[name_b]
    b_i = biases[0]
    b_f = biases[1]
    b_c = biases[2]
    b_o = biases[3]

    def _step(*args):
        inputs = args[:len(input_sizes)]
        [prev_cell,prev_hid] = args[len(input_sizes):]

        # batch_size x hidden_size
        batch_size = inputs[0].shape[0]

        # batch_size x 4 x hidden_size
        transformed_x = sum(
                T.dot(x, W).reshape((batch_size, 4, hidden_size))
                for x, W in zip(inputs,W_inputs)
            )

        # batch_size x 4 x hidden_size
        transformed_hid = T.dot(prev_hid, P[name_W_hidden]).reshape(
            (batch_size, 4, hidden_size))
        # batch_size x 2 x hidden_size
        transformed_cell = T.dot(prev_cell, V_if).reshape(
            (batch_size, 2, hidden_size))

        x_i = transformed_x[:,0]
        x_f = transformed_x[:,1]
        x_c = transformed_x[:,2]
        x_o = transformed_x[:,3]   # batch_size x hidden_size

        h_i = transformed_hid[:,0]
        h_f = transformed_hid[:,1]
        h_c = transformed_hid[:,2]
        h_o = transformed_hid[:,3]  # batch_size x hidden_size

        c_i = transformed_cell[:,0]
        c_f = transformed_cell[:,1]  # batch_size x hidden_size

        in_lin = x_i + h_i + b_i + c_i
        forget_lin = x_f + h_f + b_f + c_f
        cell_lin = x_c + h_c + b_c

        in_gate = T.nnet.sigmoid(in_lin)
        forget_gate = T.nnet.sigmoid(forget_lin)
        cell_updates = T.tanh(cell_lin)
        in_gate.name = "in_gate"
        forget_gate.name = "forget_gate"
        cell_updates.name = "cell_updates"

        cell = forget_gate * prev_cell + in_gate * cell_updates

        out_lin = x_o + h_o + b_o + T.dot(cell, V_o)
        out_gate = T.nnet.sigmoid(out_lin)
        out_gate.name = "out_gate"

        hid = out_gate * T.tanh(cell)
        return cell, hid
    return _step
