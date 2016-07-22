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

from theano.compile.nanguardmode import NanGuardMode

import model, reader

def clip(clip_size,parameters,gradients):
    grad_mag = T.sqrt(sum(T.sum(T.sqr(w)) for w in parameters))
    exploded = T.isnan(grad_mag) | T.isinf(grad_mag)
    scale = clip_size / T.maximum(clip_size,grad_mag)

    return [ T.switch(exploded,
                    0.1 * p,
                    scale * g
                ) for p,g in zip(parameters,gradients) ]

def weight_norm(u,norm=1.9356):
    in_norm = T.sqrt(T.sum(T.sqr(u),axis=0))
    ratio = T.minimum(norm,in_norm) / (in_norm + 1e-8)
    return ratio * u


def normalise_weights(updates):
    return [ (
            p,
            weight_norm(u) if p.name.startswith('W') else u
        ) for p,u in updates ]

if __name__ == "__main__":
    P = Parameters()
    extract,_ = model.build(P, "vrnn")
    X = T.tensor3('X')
    l = T.ivector('l')
    [Z_prior_mean, Z_prior_std,
        Z_mean, Z_std, X_mean, X_std] = extract(X,l)

    parameters = P.values()
    batch_cost = model.cost(X, Z_prior_mean, Z_prior_std,
                      Z_mean, Z_std, X_mean, X_std,l)
    print "Calculating gradient..."
    print parameters
    batch_size = T.cast(X.shape[1],'float32')

    gradients = T.grad(batch_cost,wrt=parameters)
#    gradients = [ g / batch_size for g in gradients ]
    gradients = clip(5,parameters,gradients)

    P_learn = Parameters()
    updates = updates.adam(parameters,gradients,learning_rate=0.001,P=P_learn)
    updates = normalise_weights(updates)

    print "Compiling..."
    train = theano.function(
            inputs=[X,l],
            outputs=batch_cost,
            updates=updates,
        )
    test = theano.function(inputs=[X,l],outputs=batch_cost)


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
            for i in xrange(1, 5)
        ])
        stream = data_io.buffered_sort(stream, key=lambda x: x[1].shape[0], buffer_items=128)
        batched_stream = reader.batch_and_pad(stream, batch_size=20, mean=mean, std=std)
        batched_stream = data_io.buffered_random(batched_stream, buffer_items=4)
        return batched_stream

    def validate():
        stream = data_io.stream_file('data/train.%02d.pklgz' % 0)
        stream = data_io.buffered_sort(stream, key=lambda x: x[1].shape[0], buffer_items=128)
        batched_stream = reader.batch_and_pad(stream, batch_size=32, mean=mean, std=std)

        total_cost = 0
        total_frames = 0
        for data, lengths in batched_stream:
            batch_avg_cost = test(data,lengths)
            batch_frames = np.sum(lengths)
            total_cost += batch_avg_cost * batch_frames
            total_frames += batch_frames
        return total_cost / total_frames

    import train_loop
    model_filename = "model.pkl.1"
    learning_filename = "learning.pkl.1"
    def save():
        P.save(model_filename)
        P_learn.save(learning_filename)
    def load():
        P.load(model_filename)
        P_learn.load(learning_filename)
    load()

    train_loop.run(
            data_iterator=stream,
            train_fun=lambda batch:train(batch[0],batch[1]),
            validation_score=validate,
            save_best_params=save,
            load_best_params=load,
            max_epochs=1000,
            patience=5000,
            patience_increase=2,
            improvement_threshold=0.999,
        )

