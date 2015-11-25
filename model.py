import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

from theano_toolkit import utils as U
from theano_toolkit import updates
from theano_toolkit.parameters import Parameters

import feedforward
import vae
import lstm


def build(P, name,
          input_size=200, z_size=200,
          hidden_layer_size=2048,
          x_extractor_layers=[512] * 4,
          z_extractor_layers=[512] * 4,
          prior_layers=[512] * 4,
          generation_layers=[512] * 4,
          inference_layers=[512] * 4):

    X_extractor = feedforward.build_classifier(
        P, "x_extractor",
        input_sizes=[input_size],
        hidden_sizes=x_extractor_layers[:-1],
        output_size=x_extractor_layers[-1],
        initial_weights=feedforward.relu_init,
        output_initial_weights=feedforward.relu_init,
        activation=T.nnet.relu,
        output_activation=T.nnet.relu
    )

    Z_extractor = feedforward.build_classifier(
        P, "z_extractor",
        input_sizes=[z_size],
        hidden_sizes=z_extractor_layers[:-1],
        output_size=z_extractor_layers[-1],
        initial_weights=feedforward.relu_init,
        output_initial_weights=feedforward.relu_init,
        activation=T.nnet.relu,
        output_activation=T.nnet.relu
    )

    prior = vae.build_inferer(
        P, "prior",
        input_sizes=[hidden_layer_size],
        hidden_sizes=prior_layers,
        output_size=z_size,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.relu,
        initialise_outputs=False
    )

    generate = vae.build_inferer(
        P, "generator",
        input_sizes=[hidden_layer_size, z_extractor_layers[-1]],
        hidden_sizes=generation_layers,
        output_size=input_size,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.relu,
        initialise_outputs=False
    )

    P.init_recurrence_hidden = np.zeros((hidden_layer_size,))
    P.init_recurrence_cell = np.zeros((hidden_layer_size,))
    recurrence = lstm.build_step(
        P, "recurrence",
        input_size=x_extractor_layers[-1] + z_extractor_layers[-1],
        hidden_size=hidden_layer_size
    )

    infer = vae.build_inferer(
        P, "infer",
        input_sizes=[hidden_layer_size, x_extractor_layers[-1]],
        hidden_sizes=generation_layers,
        output_size=z_size,
        initial_weights=feedforward.relu_init,
        activation=T.nnet.relu,
        initialise_outputs=False
    )

    def extract(X):

        init_hidden = T.tanh(P.init_recurrence_hidden)
        init_cell = P.init_recurrence_cell
        init_hidden_batch = T.alloc(init_hidden, X.shape[0], hidden_layer_size)
        init_cell_batch = T.alloc(init_cell, X.shape[0], hidden_layer_size)
        X_feat = X_extractor([X])
        def _step(x_feat, prev_cell, prev_hidden):
            _, z_prior_mean, z_prior_logvar = prior([prev_hidden])
            z_sample , z_mean, z_logvar = infer([prev_hidden, x_feat])
            z_feat = Z_extractor([z_sample])
            x_sample, x_mean, x_logvar = generate([prev_hidden, z_feat])
            curr_cell, curr_hidden = recurrence(
                    T.concatenate([
                        x_feat,
                        z_feat
                    ], axis=1),
                    prev_cell, prev_hidden
                )
            return curr_cell, curr_hidden,\
                    z_prior_mean, z_prior_logvar, \
                    z_sample,z_mean,z_logvar,\
                    x_sample,x_mean,x_logvar


        [_,_,
        Z_prior_mean,Z_prior_logvar,
        Z_sample,Z_mean,Z_logvar,
        X_sample,X_mean,X_logvar], updates = theano.scan(
                _step,
               sequences=[X_feat.dimshuffle(1,0,2)],
               outputs_info=[init_cell_batch, init_hidden_batch] +\
                            [None] * 8,
            )
        return [
                Z_prior_mean.dimshuffle(1,0,2),
                Z_prior_logvar.dimshuffle(1,0,2),
                Z_mean.dimshuffle(1,0,2),
                Z_logvar.dimshuffle(1,0,2),
                X_mean.dimshuffle(1,0,2),
                X_logvar.dimshuffle(1,0,2)
            ],updates
    return extract

def cost(X,Z_prior_mean,Z_prior_logvar,
            Z_mean,Z_logvar,X_mean,X_logvar):

    encoding_cost = vae.kl_divergence(
            mean_1=Z_prior_mean, logvar_1=Z_prior_logvar,
            mean_2=Z_mean, logvar_2=Z_logvar
        )

    reconstruction_cost = vae.gaussian_nll(X,X_mean,X_logvar)

    return -T.mean(
            T.sum(encoding_cost + reconstruction_cost,axis=-1))






if __name__ == "__main__":
    P = Parameters()
    extract = build(P, "vrnn")
    print "Compiling..."
    X = T.tensor3('X')
    [Z_prior_mean,Z_prior_logvar,
        Z_mean,Z_logvar,X_mean,X_logvar],updates = extract(X)

    batch_cost = cost(X,Z_prior_mean,Z_prior_logvar,
                        Z_mean,Z_logvar,X_mean,X_logvar)
    f = theano.function(inputs=[X],outputs=batch_cost,updates=updates)
    print "Done compiling."
    sample_input = np.random.randn(5, 100, 200).astype(np.float32)
    print f(sample_input)
