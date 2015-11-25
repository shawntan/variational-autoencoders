import theano
import theano.tensor as T
import numpy as np
from theano_toolkit import utils as U

import feedforward


def build(P, name,
          input_size,
          encoder_hidden_sizes,
          latent_size,
          decoder_hidden_sizes=None,
          activation=T.nnet.softplus,
          initial_weights=feedforward.relu_init):

    if decoder_hidden_sizes == None:
        decoder_hidden_sizes = encoder_hidden_sizes[::-1]

    encode = build_inferer(P, "%s_encoder" % name,
                           [input_size],
                           encoder_hidden_sizes,
                           latent_size,
                           activation=activation,
                           initial_weights=initial_weights,
                           initialise_outputs=True
                           )
    decode = build_inferer(P, "%s_decoder" % name,
                           [latent_size],
                           decoder_hidden_sizes,
                           input_size,
                           activation=activation,
                           initial_weights=initial_weights)

    def recon_error(X, encode=encode, decode=decode):
        Z_latent, Z_mean, Z_logvar = encode([X])
        _, recon_X_mean, recon_X_logvar = decode([Z_latent])
        KL_d = kl_divergence(Z_mean, Z_logvar)
        log_p_X = gaussian_log(recon_X_mean, recon_X_logvar, X)
        cost = -(log_p_X - KL_d)
        return recon_X_mean, T.mean(cost), T.mean(KL_d), T.mean(log_p_X)
    return encode, decode, recon_error


def gaussian_nll(X, mean, logvar):
    return - 0.5 * T.sum(
        np.log(2 * np.pi) + logvar +
        T.sqr(X - mean) / T.exp(logvar), axis=-1)


def kl_divergence(mean_1, logvar_1, mean_2, logvar_2):
    return - 0.5 * T.sum(
        logvar_2 - logvar_1 +
        ((T.exp(logvar_1) + T.sqr(mean_1 - mean_2))
         / T.exp(logvar_2)), axis=-1
    )


def build_inferer(P, name, input_sizes, hidden_sizes, output_size,
                  initial_weights, activation,
                  initialise_outputs=False):

    combine_inputs = feedforward.build_combine_transform(
        P, "%s_input" % name,
        input_sizes, hidden_sizes[0],
        initial_weights=initial_weights,
        activation=activation
    )

    transform = feedforward.build_stacked_transforms(
        P, name, hidden_sizes,
        initial_weights=initial_weights,
        activation=activation)

    output = build_encoder_output(
        P, name,
        hidden_sizes[-1], output_size,
        initialise_weights=(initial_weights if initialise_outputs else None)
    )

    def infer(Xs, samples=-1):
        combine = combine_inputs(Xs)
        hiddens = transform(combine)
        latent, mean, logvar = output(hiddens[-1], samples=samples)
        return latent, mean, logvar
    return infer


def build_encoder_output(P, name, input_size, output_size, initialise_weights=None):

    if initialise_weights is None:
        initialise_weights = lambda x, y: np.zeros((x, y))

    P["W_%s_mean" % name] = 0.01 * np.random.randn(input_size, output_size)
    P["b_%s_mean" % name] = np.zeros((output_size,))
    P["W_%s_logvar" % name] = np.zeros((input_size, output_size))
    P["b_%s_logvar" % name] = np.zeros((output_size,))

    def output(X, samples=-1):
        mean = T.dot(X, P["W_%s_mean" % name]) + P["b_%s_mean" % name]
        logvar = T.dot(X, P["W_%s_logvar" % name]) + P["b_%s_logvar" % name]

        std = T.exp(0.5 * logvar)
        if samples == -1:
            eps = U.theano_rng.normal(size=(logvar.shape[0], output_size))
        else:
            eps = U.theano_rng.normal(size=(logvar.shape[0], samples, output_size))
            std = std.dimshuffle(0, 'x', 1)
            mean = mean.dimshuffle(0, 'x', 1)

        latent = mean + eps * std
        return latent, mean, logvar
    return output
