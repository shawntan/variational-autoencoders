import theano
import theano.tensor as T
import numpy as np
import cPickle as pickle

from theano_toolkit import utils as U
from theano_toolkit.parameters import Parameters

import feedforward
import vae
import lstm


def build(P, name,
          input_size=200, z_size=200,
          hidden_layer_size=2500,
          x_extractor_layers=[600] * 4,
          z_extractor_layers=[500] * 4,
          prior_layers=[500] * 4,
          generation_layers=[600] * 4,
          inference_layers=[500] * 4):

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
        init_hidden_batch = T.alloc(init_hidden, X.shape[1], hidden_layer_size)
        init_cell_batch = T.alloc(init_cell, X.shape[1], hidden_layer_size)
        noise = U.theano_rng.normal(size=(X.shape[0],X.shape[1],z_size))
        reset_init_mask = U.theano_rng.binomial(size=(X.shape[0],X.shape[1]),p=0.01)

        X_feat = X_extractor([X])

        def _step(x_feat, eps, reset_mask, prev_cell, prev_hidden):
            reset_mask = reset_mask.dimshuffle(0,'x')
            prev_cell = T.switch(
                    reset_mask, init_cell_batch, prev_cell)

            prev_hidden = T.switch(
                    reset_mask, init_hidden_batch, prev_hidden)


            _, z_prior_mean, z_prior_logvar = prior([prev_hidden])
            _, z_mean, z_logvar = infer([prev_hidden, x_feat])
            z_sample = z_mean + eps * T.exp(0.5 * z_logvar)
            z_feat = Z_extractor([z_sample])
            _, x_mean, x_logvar = generate([prev_hidden, z_feat])

            curr_cell, curr_hidden = recurrence(
                T.concatenate([
                    x_feat,
                    z_feat
                ], axis=1),
                prev_cell, prev_hidden
            )
            return curr_cell, curr_hidden,\
                z_prior_mean, z_prior_logvar, \
                z_sample, z_mean, z_logvar,\
                x_mean, x_logvar

        [_, _,
         Z_prior_mean, Z_prior_logvar,
         Z_sample, Z_mean, Z_logvar,
         X_mean, X_logvar], _ = theano.scan(
            _step,
            sequences=[X_feat,noise,reset_init_mask],
            outputs_info=[init_cell_batch, init_hidden_batch] +
            [None] * 7,
        )
        return [
            Z_prior_mean, Z_prior_logvar,
            Z_mean, Z_logvar,
            X_mean, X_logvar,
        ]
    return extract


def cost(X, Z_prior_mean, Z_prior_logvar,
         Z_mean, Z_logvar, X_mean, X_logvar,
         lengths):
    mask = T.arange(X.shape[0]).dimshuffle(0,'x')\
            < lengths.dimshuffle('x',0)

    encoding_cost = mask * vae.kl_divergence(
        mean_1=Z_prior_mean, logvar_1=Z_prior_logvar,
        mean_2=Z_mean, logvar_2=Z_logvar
    )

    reconstruction_cost = mask * vae.gaussian_nll(X, X_mean, X_logvar)

    return -T.sum(encoding_cost + reconstruction_cost)/T.sum(mask)



