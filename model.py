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
    def weight_init(x, y):
        return np.random.uniform(-0.08, 0.08, (x, y))

    X_extractor = feedforward.build_classifier(
        P, "x_extractor",
        input_sizes=[input_size],
        hidden_sizes=x_extractor_layers[:-1],
        output_size=x_extractor_layers[-1],
        initial_weights=weight_init,
        output_initial_weights=weight_init,
        activation=T.nnet.relu,
        output_activation=T.nnet.relu
    )

    Z_extractor = feedforward.build_classifier(
        P, "z_extractor",
        input_sizes=[z_size],
        hidden_sizes=z_extractor_layers[:-1],
        output_size=z_extractor_layers[-1],
        initial_weights=weight_init,
        output_initial_weights=weight_init,
        activation=T.nnet.relu,
        output_activation=T.nnet.relu
    )

    prior = vae.build_inferer(
        P, "prior",
        input_sizes=[hidden_layer_size],
        hidden_sizes=prior_layers,
        output_size=z_size,
        initial_weights=weight_init,
        activation=T.nnet.relu,
        initialise_outputs=True
    )

    generate = vae.build_inferer(
        P, "generator",
        input_sizes=[hidden_layer_size, z_extractor_layers[-1]],
        hidden_sizes=generation_layers,
        output_size=input_size,
        initial_weights=weight_init,
        activation=T.nnet.relu,
        initialise_outputs=True
    )

    P.init_recurrence_hidden = np.zeros((hidden_layer_size,))
    P.init_recurrence_cell = np.zeros((hidden_layer_size,))
    recurrence = lstm.build_step(
        P, "recurrence",
        input_sizes=[x_extractor_layers[-1], z_extractor_layers[-1]],
        hidden_size=hidden_layer_size
    )

    infer = vae.build_inferer(
        P, "infer",
        input_sizes=[hidden_layer_size, x_extractor_layers[-1]],
        hidden_sizes=generation_layers,
        output_size=z_size,
        initial_weights=weight_init,
        activation=T.nnet.relu,
        initialise_outputs=True
    )

    def sample():
        init_hidden = T.tanh(P.init_recurrence_hidden)
        init_cell = P.init_recurrence_cell
        init_hidden_batch = T.alloc(init_hidden, 1, hidden_layer_size)
        init_cell_batch = T.alloc(init_cell, 1, hidden_layer_size)
        noise = U.theano_rng.normal(size=(160, 1, z_size))

        def _step(eps, prev_cell, prev_hidden):
            _, z_prior_mean, z_prior_std = prior([prev_hidden])
            z_sample = z_prior_mean + eps * z_prior_std
            z_feat = Z_extractor([z_sample])
            _, x_mean, _ = generate([prev_hidden, z_feat])
            x_feat = X_extractor([x_mean])
            curr_cell, curr_hidden = recurrence(x_feat, z_feat, prev_cell, prev_hidden)
            return curr_cell, curr_hidden, x_mean

        [cells, hiddens, x_means], _ = theano.scan(
            _step,
            sequences=[noise],
            outputs_info=[init_cell_batch, init_hidden_batch, None],
        )
        return x_means

    def extract(X, l):

        init_hidden = T.tanh(P.init_recurrence_hidden)
        init_cell = P.init_recurrence_cell
        init_hidden_batch = T.alloc(init_hidden, X.shape[1], hidden_layer_size)
        init_cell_batch = T.alloc(init_cell, X.shape[1], hidden_layer_size)
        noise = U.theano_rng.normal(size=(X.shape[0], X.shape[1], z_size))
        reset_init_mask = U.theano_rng.binomial(size=(X.shape[0], X.shape[1]), p=0.025)

        X_feat = X_extractor([X])

        def _step(t, x_feat, eps, reset_mask, prev_cell, prev_hidden):
            reset_mask = reset_mask.dimshuffle(0, 'x')

            _, z_prior_mean, z_prior_std = prior([prev_hidden])
            _, z_mean, z_std = infer([prev_hidden, x_feat])
            z_sample = z_mean + eps * z_std
            z_feat = Z_extractor([z_sample])

            curr_cell, curr_hidden = recurrence(
                x_feat, z_feat,
                prev_cell, prev_hidden
            )

            curr_cell = T.switch(
                reset_mask, init_cell_batch, curr_cell)
            curr_hidden = T.switch(
                reset_mask, init_hidden_batch, curr_hidden)
            mask = (t < l).dimshuffle(0, 'x')

            return tuple(
                T.switch(mask, out, 0)
                for out in (
                    curr_cell, curr_hidden,
                    z_prior_mean, z_prior_std,
                    z_sample, z_mean, z_std, z_feat
                ))

        [_, hiddens,
         Z_prior_mean, Z_prior_std,
         Z_sample, Z_mean, Z_std, Z_feats], _ = theano.scan(
            _step,
            sequences=[T.arange(X_feat.shape[0]), X_feat, noise, reset_init_mask],
            outputs_info=[init_cell_batch, init_hidden_batch] + [None] * 6,
        )

        hiddens = T.concatenate([
            init_hidden_batch.dimshuffle('x', 0, 1),
            hiddens[:-1]
        ])

        _, X_mean, X_std = generate([hiddens, Z_feats])

        return [
            Z_prior_mean, Z_prior_std,
            Z_mean, Z_std,
            X_mean, X_std,
        ]
    return extract, sample


def cost(X,
         Z_prior_mean, Z_prior_std,
         Z_mean, Z_std,
         X_mean, X_std,
         lengths):
    mask = T.arange(X.shape[0]).dimshuffle(0, 'x')\
        < lengths.dimshuffle('x', 0)

    encoding_cost = T.switch(mask,
                             vae.kl_divergence(
                                 mean_1=Z_mean, std_1=Z_std,
                                 mean_2=Z_prior_mean, std_2=Z_prior_std,
                             ),
                             0
                             )

    reconstruction_cost = T.switch(mask,
                                   vae.gaussian_nll(X, X_mean, X_std),
                                   0
                                   )

    return (-T.sum(encoding_cost) -T.sum(reconstruction_cost)) / T.sum(mask)
