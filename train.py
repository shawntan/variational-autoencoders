import numpy as np
import cPickle as pickle
import gzip
import data_io
import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters


import model, reader

def clip(clip_size,gradients):
    grad_mag = T.sqrt(sum(T.sum(T.sqr(w)) for w in parameters))
    scale = clip_size / T.maximum(clip_size,grad_mag)
    return [ scale * g for g in gradients ]

def nan_avoid(parameters,gradients):
    return [ T.switch(T.isnan(g) | T.isinf(g),0.05 * p, g)
                for p, g in zip(parameters,gradients) ]

def weight_norm(u,norm=1.9356):
    if u.ndim == 2:
        in_norm = T.sqrt(T.sum(T.sqr(u),axis=0))
        ratio = T.minimum(norm,in_norm) / (in_norm + 1e-8)
        return ratio * u
    else:
        return u


def normalise_weights(updates):
    return [ (p,weight_norm(u)) for p,u in updates ]

if __name__ == "__main__":
    P = Parameters()
    extract = model.build(P, "vrnn")
    X = T.tensor3('X')
    l = T.ivector('l')
    [Z_prior_mean, Z_prior_logvar,
        Z_mean, Z_logvar, X_mean, X_logvar] = extract(X)

    parameters = P.values()
    batch_cost = model.cost(X, Z_prior_mean, Z_prior_logvar,
                      Z_mean, Z_logvar, X_mean, X_logvar,l)
    print "Calculating gradient..."
    print parameters
    gradients = T.grad(batch_cost,wrt=parameters)
    batch_size = T.cast(X.shape[1],'float32')
    gradients = clip(5,gradients)
    gradients = nan_avoid(parameters,gradients)

    P_learn = Parameters()
    updates = updates.adam(parameters,gradients,P=P_learn)
    updates = normalise_weights(updates)

    print "Compiling..."
    train = theano.function(
            inputs=[X,l],
            outputs=batch_cost,
            updates=updates
        )

    print "Calculating mean variance..."
    rand_stream = data_io.random_select_stream(*[
        data_io.stream_file('data/train.%02d.pklgz' % i)
        for i in xrange(1, 20)
    ])

    mean, std, count = reader.get_normalise(rand_stream)
    print "Dataset count:", count

    def stream():
        stream = data_io.random_select_stream(*[
            data_io.stream_file('data/train.%02d.pklgz' % i)
            for i in xrange(1, 20)
        ])
        stream = data_io.buffered_sort(stream, key=lambda x: x[1].shape[0], buffer_items=128)
        batched_stream = reader.batch_and_pad(stream, batch_size=16, mean=mean, std=std)
        batched_stream = data_io.buffered_random(batched_stream, buffer_items=4)
        return batched_stream

    print "Training..."
#    P.load('model.pkl')
#    P_learn.load('model.lrn.pkl')
    for epoch in xrange(10):
        print "New epoch"
        for data, lengths in stream():
            print lengths
            print train(data,lengths)
        P.save('model.pkl')
        P_learn.save('model.lrn.pkl')
