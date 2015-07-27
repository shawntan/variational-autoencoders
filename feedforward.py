import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters


def build(P, name,
          input_sizes, hidden_sizes, output_size,
          activation=T.tanh):

    input_weights = []
    for i, size in enumerate(input_sizes):
        P["W_%s_hidden_input_%d" % (name, i)] = \
            (0.1 * 2) * (np.random.rand(size, hidden_sizes[0]) - 0.5)
        input_weights.append(P["W_%s_hidden_input_%d" % (name, i)])
    P["b_%s_hidden_input" % name] = np.zeros((hidden_sizes[0],), dtype=np.float32)
    b_hidden_input = P["b_%s_hidden_input" % name]
    hidden_weights = []
    prev_size = hidden_sizes[0]
    for i, size in enumerate(hidden_sizes[1:]):
        P["W_%s_hidden_%d" % (name, i)] = (0.1 * 2) * (np.random.rand(prev_size, size) - 0.5)
        P["b_%s_hidden_%d" % (name, i)] = np.zeros((size,), dtype=np.float32)
        prev_size = size
        hidden_weights.append((P["W_%s_hidden_%d" % (name, i)], P["b_%s_hidden_%d" % (name, i)]))

    P["W_%s_output" % name] = np.zeros((prev_size, output_size), dtype=np.float32)
    P["b_%s_output" % name] = np.zeros((output_size,), dtype=np.float32)
    W_output = P["W_%s_output" % name]
    b_output = P["b_%s_output" % name]

    def feedforward(X):
        lin_hidden = sum(T.dot(x, input_weight) for x, input_weight in zip(X, input_weights))
        hidden = activation(lin_hidden + b_hidden_input)
        for W, b in hidden_weights:
            hidden = activation(T.dot(hidden, W) + b)
        output = T.dot(hidden, W_output) + b_output
        return output
    return feedforward
