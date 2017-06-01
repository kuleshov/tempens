import time
import pickle
import numpy as np

import theano
import theano.tensor as T
import lasagne

from lasagne.layers import batch_norm
from layers.sampling import GaussianSampleLayer
from layers.shape import RepeatLayer
from layers import extras as nn

from distributions import log_bernoulli, log_normal, log_normal2

from zca_bn import mean_only_bn as WN

# ----------------------------------------------------------------------------

conv_defs = {
        'W': lasagne.init.HeNormal('relu'),
        'b': lasagne.init.Constant(0.0),
        'filter_size': (3, 3),
        'stride': (1, 1),
        'nonlinearity': lasagne.nonlinearities.LeakyRectify(0.1)
    }

nin_defs = {
    'W': lasagne.init.HeNormal('relu'),
    'b': lasagne.init.Constant(0.0),
    'nonlinearity': lasagne.nonlinearities.LeakyRectify(0.1)
}

dense_defs = {
    'W': lasagne.init.HeNormal(1.0),
    'b': lasagne.init.Constant(0.0),
    'nonlinearity': lasagne.nonlinearities.softmax
}

wn_defs = {
    'momentum': 0.999
}

# ----------------------------------------------------------------------------    

def create_model0(X, n_dim, n_out, n_chan=1, model='bernoulli'):
    conv_defs = {
        'W': lasagne.init.HeNormal('relu'),
        'b': lasagne.init.Constant(0.0),
        'filter_size': (3, 3),
        'stride': (1, 1),
        'nonlinearity': lasagne.nonlinearities.LeakyRectify(0.1)
    }

    nin_defs = {
        'W': lasagne.init.HeNormal('relu'),
        'b': lasagne.init.Constant(0.0),
        'nonlinearity': lasagne.nonlinearities.LeakyRectify(0.1)
    }

    dense_defs = {
        'W': lasagne.init.HeNormal(1.0),
        'b': lasagne.init.Constant(0.0),
        'nonlinearity': lasagne.nonlinearities.softmax
    }

    wn_defs = {
        'momentum': 0.999
    }

    net = lasagne.layers.InputLayer        (     name='input',    shape=(None, n_chan, n_dim, n_dim), input_var=X)
    net = lasagne.layers.DropoutLayer(net, p=0.2)

    net = lasagne.layers.Conv2DLayer(
        net, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.GlorotUniform())
    net = lasagne.layers.MaxPool2DLayer(
        net, pool_size=(3, 3), stride=(2,2))

    net = lasagne.layers.Conv2DLayer(
        net, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.GlorotUniform())
    net = lasagne.layers.MaxPool2DLayer(
        net, pool_size=(3, 3), stride=(2,2))

    net = lasagne.layers.Conv2DLayer(
        net, num_filters=128, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.GlorotUniform())
    net = lasagne.layers.MaxPool2DLayer(
        net, pool_size=(3, 3), stride=(2,2))

    net = lasagne.layers.DenseLayer(
        net, num_units=1000,
        nonlinearity=lasagne.nonlinearities.rectify)
    net = lasagne.layers.DropoutLayer(net, p=0.5)

    net = lasagne.layers.DenseLayer(
            net, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return net

def create_model(X, n_dim, n_out, n_chan=1, model='bernoulli'):
    # params
    n_lat = 200 # latent stochastic variables
    n_aux = 10  # auxiliary variables
    n_hid = 499 # size of hidden layer in encoder/decoder
    n_sam = 1 # number of monte-carlo samples
    n_out = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl = lasagne.nonlinearities.rectify
    relu_shift = lambda av: T.nnet.relu(av+10)-10 # for numerical stability

    # create the encoder network

    # create q(a|x)
    l_qa_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    l_qa_hid1 = (lasagne.layers.DenseLayer(
        l_qa_in, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_qa_mu = lasagne.layers.DenseLayer(
        l_qa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_qa_logsigma = lasagne.layers.DenseLayer(
        l_qa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)
    l_qa_mu = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_mu, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa_logsigma = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_logsigma, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa = GaussianSampleLayer(l_qa_mu, l_qa_logsigma)

    # create q(z|a,x)
    l_qz_hid1a = (lasagne.layers.DenseLayer(
        l_qa, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_qz_hid1b = (lasagne.layers.DenseLayer(
        l_qa_in, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_qz_hid1b = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qz_hid1b, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_qz_hid2 = lasagne.layers.ElemwiseSumLayer([l_qz_hid1a, l_qz_hid1b])
    l_qz_hid2 = lasagne.layers.NonlinearityLayer(l_qz_hid2, hid_nl)
    l_qz_mu = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_qz_logsigma = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)
    l_qz = GaussianSampleLayer(l_qz_mu, l_qz_logsigma, name='l_qz')

    # create the decoder network

    # create p(x|z)
    l_px_hid1 = (lasagne.layers.DenseLayer(
        l_qz, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_px_mu, l_px_logsigma = None, None

    if model == 'bernoulli':
      l_px_mu = lasagne.layers.DenseLayer(l_px_hid1, num_units=n_out,
          nonlinearity = lasagne.nonlinearities.sigmoid,
          W=lasagne.init.Orthogonal(),
          b=lasagne.init.Constant(0.0))
    elif model == 'gaussian':
      l_px_mu = lasagne.layers.DenseLayer(
          l_px_hid1, num_units=n_out,
          W=lasagne.init.Orthogonal(),
          b=lasagne.init.Constant(0.0),
          nonlinearity=None)
      l_px_logsigma = lasagne.layers.DenseLayer(
          l_px_hid1, num_units=n_out,
          W=lasagne.init.Orthogonal(),
          b=lasagne.init.Constant(0.0),
          nonlinearity=relu_shift)

    # create p(a|z)
    l_pa_hid1 = (lasagne.layers.DenseLayer(
      l_qz, num_units=n_hid,
      nonlinearity=hid_nl,
      W=lasagne.init.Orthogonal(),
      b=lasagne.init.Constant(0.0)))
    l_pa_mu = lasagne.layers.DenseLayer(
        l_pa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_pa_logsigma = lasagne.layers.DenseLayer(
        l_pa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)

    # discriminative model
    l_in_drop = lasagne.layers.DropoutLayer(l_qa_in, p=0.2)
    # l_in_drop = l_in

    l_conv1 = lasagne.layers.Conv2DLayer(
        l_in_drop, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.Orthogonal())
    l_conv1 = lasagne.layers.MaxPool2DLayer(
        l_conv1, pool_size=(3, 3), stride=(2,2))

    l_conv2 = lasagne.layers.Conv2DLayer(
        l_conv1, num_filters=64, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.Orthogonal())
    l_conv2 = lasagne.layers.MaxPool2DLayer(
        l_conv2, pool_size=(3, 3), stride=(2,2))

    l_conv3 = lasagne.layers.Conv2DLayer(
        l_conv2, num_filters=128, filter_size=(5, 5),
        nonlinearity=lasagne.nonlinearities.rectify,
        pad='same', W=lasagne.init.Orthogonal())
    l_conv3 = lasagne.layers.MaxPool2DLayer(
        l_conv3, pool_size=(3, 3), stride=(2,2))
    l_conv3 = lasagne.layers.FlattenLayer(l_conv3)

    l_merge = lasagne.layers.ConcatLayer([l_conv3, l_qz_mu])

    l_hid = lasagne.layers.DenseLayer(
        l_merge, num_units=1000,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=lasagne.nonlinearities.rectify)
    l_hid_drop = lasagne.layers.DropoutLayer(l_hid, p=0.5)
    # l_hid_drop = l_hid

    l_d = lasagne.layers.DenseLayer(
            l_hid_drop, num_units=10,
            nonlinearity=lasagne.nonlinearities.softmax)

    return l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
           l_qz_mu, l_qz_logsigma, l_qa_mu, l_qa_logsigma, \
           l_qa, l_qz, l_d

def create_model2(X, n_dim, n_out, n_chan=1, model='gaussian'):
    # params
    n_lat = 200 # latent stochastic variables
    n_aux = 10  # auxiliary variables
    n_hid = 499 # size of hidden layer in encoder/decoder
    n_sam = 1 # number of monte-carlo samples
    n_out = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl = lasagne.nonlinearities.rectify
    relu_shift = lambda av: T.nnet.relu(av+10)-10 # for numerical stability

    # create the encoder network

    # create q(a|x)
    l_qa_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)
    l_qa_hid1 = (lasagne.layers.DenseLayer(
        l_qa_in, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_qa_mu = lasagne.layers.DenseLayer(
        l_qa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_qa_logsigma = lasagne.layers.DenseLayer(
        l_qa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)
    l_qa_mu = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_mu, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa_logsigma = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qa_logsigma, n_ax=1, n_rep=n_sam),
        shape=(-1, n_aux))
    l_qa = GaussianSampleLayer(l_qa_mu, l_qa_logsigma)

    # create q(z|a,x)
    l_qz_hid1a = (lasagne.layers.DenseLayer(
        l_qa, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_qz_hid1b = (lasagne.layers.DenseLayer(
        l_qa_in, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_qz_hid1b = lasagne.layers.ReshapeLayer(
        RepeatLayer(l_qz_hid1b, n_ax=1, n_rep=n_sam),
        shape=(-1, n_hid))
    l_qz_hid2 = lasagne.layers.ElemwiseSumLayer([l_qz_hid1a, l_qz_hid1b])
    l_qz_hid2 = lasagne.layers.NonlinearityLayer(l_qz_hid2, hid_nl)
    l_qz_mu = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_qz_logsigma = lasagne.layers.DenseLayer(
        l_qz_hid2, num_units=n_lat,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)
    l_qz = GaussianSampleLayer(l_qz_mu, l_qz_logsigma, name='l_qz')

    # create the decoder network

    # create p(x|z)
    l_px_hid1 = (lasagne.layers.DenseLayer(
        l_qz, num_units=n_hid,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=hid_nl))
    l_px_mu, l_px_logsigma = None, None

    if model == 'bernoulli':
      l_px_mu = lasagne.layers.DenseLayer(l_px_hid1, num_units=n_out,
          nonlinearity = lasagne.nonlinearities.sigmoid,
          W=lasagne.init.Orthogonal(),
          b=lasagne.init.Constant(0.0))
    elif model == 'gaussian':
      l_px_mu = lasagne.layers.DenseLayer(
          l_px_hid1, num_units=n_out,
          W=lasagne.init.Orthogonal(),
          b=lasagne.init.Constant(0.0),
          nonlinearity=None)
      l_px_logsigma = lasagne.layers.DenseLayer(
          l_px_hid1, num_units=n_out,
          W=lasagne.init.Orthogonal(),
          b=lasagne.init.Constant(0.0),
          nonlinearity=relu_shift)

    # create p(a|z)
    l_pa_hid1 = (lasagne.layers.DenseLayer(
      l_qz, num_units=n_hid,
      nonlinearity=hid_nl,
      W=lasagne.init.Orthogonal(),
      b=lasagne.init.Constant(0.0)))
    l_pa_mu = lasagne.layers.DenseLayer(
        l_pa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_pa_logsigma = lasagne.layers.DenseLayer(
        l_pa_hid1, num_units=n_aux,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)

    net = lasagne.layers.GaussianNoiseLayer(l_qa_in, name='noise',    sigma=0.15)
    net = WN(lasagne.layers.Conv2DLayer    (net, name='conv1a',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    net = WN(lasagne.layers.Conv2DLayer    (net, name='conv1b',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    net = WN(lasagne.layers.Conv2DLayer    (net, name='conv1c',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    net = lasagne.layers.MaxPool2DLayer    (net, name='pool1',    pool_size=(2, 2))
    net = lasagne.layers.DropoutLayer      (net, name='drop1',    p=.5)
    net = WN(lasagne.layers.Conv2DLayer    (net, name='conv2a',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    net = WN(lasagne.layers.Conv2DLayer    (net, name='conv2b',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    net = WN(lasagne.layers.Conv2DLayer    (net, name='conv2c',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    net = lasagne.layers.MaxPool2DLayer    (net, name='pool2',    pool_size=(2, 2))
    net = lasagne.layers.DropoutLayer      (net, name='drop2',    p=.5)
    net = WN(lasagne.layers.Conv2DLayer    (net, name='conv3a',   num_filters=512, pad=0,      **conv_defs), **wn_defs)
    net = WN(lasagne.layers.NINLayer       (net, name='conv3b',   num_units=256,               **nin_defs),  **wn_defs)
    net = WN(lasagne.layers.NINLayer       (net, name='conv3c',   num_units=128,               **nin_defs),  **wn_defs)
    net = lasagne.layers.GlobalPoolLayer   (net, name='pool3')    

    net = lasagne.layers.ConcatLayer([net, l_qz_mu])

    l_d = WN(lasagne.layers.DenseLayer     (net, name='dense',    num_units=10,       **dense_defs), **wn_defs)

    return l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
           l_qz_mu, l_qz_logsigma, l_qa_mu, l_qa_logsigma, \
           l_qa, l_qz, l_d

def create_gen_objective(X, network, deterministic=False, model='bernoulli'):
    # load network input
    x = X.flatten(2)

    # duplicate entries to take into account multiple mc samples
    n_sam = 1
    n_out = x.shape[1]
    x = x.dimshuffle(0,'x',1).repeat(n_sam, axis=1).reshape((-1, n_out))

    # load network
    l_px_mu, l_px_logsigma, l_pa_mu, l_pa_logsigma, \
      l_qz_mu, l_qz_logsigma, l_qa_mu, l_qa_logsigma, \
      l_qa, l_qz, l_d = network
    
    # load network output
    pa_mu, pa_logsigma, qz_mu, qz_logsigma, qa_mu, qa_logsigma, a, z \
      = lasagne.layers.get_output(
          [ l_pa_mu, l_pa_logsigma, l_qz_mu, l_qz_logsigma, 
            l_qa_mu, l_qa_logsigma, l_qa, l_qz ], 
          deterministic=deterministic)

    if model == 'bernoulli':
      px_mu = lasagne.layers.get_output(l_px_mu, deterministic=deterministic)
    elif model == 'gaussian':
      px_mu, px_logsigma = lasagne.layers.get_output([l_px_mu, l_px_logsigma], 
                                                     deterministic=deterministic)

    # entropy term
    log_qa_given_x  = log_normal2(a, qa_mu, qa_logsigma).sum(axis=1)
    log_qz_given_ax = log_normal2(z, qz_mu, qz_logsigma).sum(axis=1)
    log_qza_given_x = log_qz_given_ax + log_qa_given_x

    # log-probability term
    z_prior_sigma = T.cast(T.ones_like(qz_logsigma), dtype=theano.config.floatX)
    z_prior_mu = T.cast(T.zeros_like(qz_mu), dtype=theano.config.floatX)
    log_pz = log_normal(z, z_prior_mu,  z_prior_sigma).sum(axis=1)
    log_pa_given_z = log_normal2(a, pa_mu, pa_logsigma).sum(axis=1)

    if model == 'bernoulli':
      log_px_given_z = log_bernoulli(x, px_mu).sum(axis=1)
    elif model == 'gaussian':
      log_px_given_z = log_normal2(x, px_mu, px_logsigma).sum(axis=1)

    log_paxz = log_pa_given_z + log_px_given_z + log_pz

    # discriminative component
    # P = lasagne.layers.get_output(l_d)
    # P_test = lasagne.layers.get_output(l_d, deterministic=True)
    # disc_loss = lasagne.objectives.categorical_crossentropy(P, Y)

    # compute the evidence lower bound
    # elbo = T.mean(-disc_loss + log_paxz - log_qza_given_x)
    elbo = T.mean(log_paxz - log_qza_given_x)
    # elbo = T.mean(-disc_loss)

    return -elbo

def get_params(network):
    l_px_mu = network[0]
    l_pa_mu = network[2]
    l_d = network[-1]
    params  = lasagne.layers.get_all_params([l_px_mu, l_pa_mu, l_d], trainable=True)
    params  = lasagne.layers.get_all_params(network, trainable=True)
    # params1 = lasagne.layers.get_all_params(l_d, trainable=True)
    
    return params

# ----------------------------------------------------------------------------
# helper layers

class Deconv2DLayer(lasagne.layers.Layer):

    def __init__(self, incoming, num_filters, filter_size, stride=1, pad=0,
            nonlinearity=lasagne.nonlinearities.rectify, **kwargs):
        super(Deconv2DLayer, self).__init__(incoming, **kwargs)
        self.num_filters = num_filters
        self.filter_size = lasagne.utils.as_tuple(filter_size, 2, int)
        self.stride = lasagne.utils.as_tuple(stride, 2, int)
        self.pad = lasagne.utils.as_tuple(pad, 2, int)
        self.W = self.add_param(lasagne.init.Orthogonal(),
                (self.input_shape[1], num_filters) + self.filter_size,
                name='W')
        self.b = self.add_param(lasagne.init.Constant(0),
                (num_filters,),
                name='b')
        if nonlinearity is None:
            nonlinearity = lasagne.nonlinearities.identity
        self.nonlinearity = nonlinearity

    def get_output_shape_for(self, input_shape):
        shape = tuple(i*s - 2*p + f - 1
                for i, s, p, f in zip(input_shape[2:],
                                      self.stride,
                                      self.pad,
                                      self.filter_size))
        return (input_shape[0], self.num_filters) + shape

    def get_output_for(self, input, **kwargs):
        op = T.nnet.abstract_conv.AbstractConv2d_gradInputs(
            imshp=self.output_shape,
            kshp=(self.input_shape[1], self.num_filters) + self.filter_size,
            subsample=self.stride, border_mode=self.pad)
        conved = op(self.W, input, self.output_shape[2:])
        if self.b is not None:
            conved += self.b.dimshuffle('x', 0, 'x', 'x')
        return self.nonlinearity(conved)        

# ----------------------------------------------------------------------------
# probably not useful

def create_model3(X, n_dim, n_out, n_chan=1, model='gaussian'):
    """Large disc with VAE gen"""
    # params
    n_lat = 200 # latent stochastic variables
    n_aux = 10  # auxiliary variables
    n_hid = 499 # size of hidden layer in encoder/decoder
    n_out = n_dim * n_dim * n_chan # total dimensionality of ouput
    hid_nl = lasagne.nonlinearities.rectify
    relu_shift = lambda av: T.nnet.relu(av+10)-10 # for numerical stability

    # create the encoder network q(z|x)
    l_q_in = lasagne.layers.InputLayer(shape=(None, n_chan, n_dim, n_dim), 
                                     input_var=X)

    # discriminative model
    from zca_bn import mean_only_bn as WN
  
    enc = lasagne.layers.GaussianNoiseLayer(l_q_in, name='noise',    sigma=0.15)
    enc = WN(lasagne.layers.Conv2DLayer    (enc, name='conv1a',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    enc = WN(lasagne.layers.Conv2DLayer    (enc, name='conv1b',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    enc = WN(lasagne.layers.Conv2DLayer    (enc, name='conv1c',   num_filters=128, pad='same', **conv_defs), **wn_defs)
    enc = lasagne.layers.MaxPool2DLayer    (enc, name='pool1',    pool_size=(2, 2))
    enc = lasagne.layers.DropoutLayer      (enc, name='drop1',    p=.5)
    enc = WN(lasagne.layers.Conv2DLayer    (enc, name='conv2a',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    enc = WN(lasagne.layers.Conv2DLayer    (enc, name='conv2b',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    enc = WN(lasagne.layers.Conv2DLayer    (enc, name='conv2c',   num_filters=256, pad='same', **conv_defs), **wn_defs)
    enc = lasagne.layers.MaxPool2DLayer    (enc, name='pool2',    pool_size=(2, 2))
    enc = lasagne.layers.DropoutLayer      (enc, name='drop2',    p=.5)
    enc = WN(lasagne.layers.Conv2DLayer    (enc, name='conv3a',   num_filters=512, pad=0,      **conv_defs), **wn_defs)
    enc = WN(lasagne.layers.NINLayer       (enc, name='conv3b',   num_units=256,               **nin_defs),  **wn_defs)
    enc = WN(lasagne.layers.NINLayer       (enc, name='conv3c',   num_units=128,               **nin_defs),  **wn_defs)
    enc = lasagne.layers.GlobalPoolLayer   (enc, name='pool3')    

    l_q_mu = lasagne.layers.DenseLayer(
        enc, num_units=n_lat,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=None)
    l_q_logsigma = lasagne.layers.DenseLayer(
        enc, num_units=n_lat,
        W=lasagne.init.Orthogonal(),
        b=lasagne.init.Constant(0.0),
        nonlinearity=relu_shift)
    l_p_z = GaussianSampleLayer(l_q_mu, l_q_logsigma, name='l_qz')

    l_d = WN(lasagne.layers.DenseLayer     (l_q_mu, name='dense',    num_units=10,       **dense_defs), **wn_defs)

    # create the decoder network
    # create p(x|z)

    from lasagne.layers import batch_norm
    l_g_hid2 = batch_norm(lasagne.layers.DenseLayer(l_p_z, 256*4*4))
    l_g_hid2 = lasagne.layers.ReshapeLayer(l_g_hid2, ([0], 256, 4, 4))
    l_g_dc1 = batch_norm(Deconv2DLayer(l_g_hid2, 128, 5, stride=2, pad=2))
    l_g_dc2 = batch_norm(Deconv2DLayer(l_g_dc1, 64, 5, stride=2, pad=2))

    if model == 'bernoulli':
      gen = Deconv2DLayer(l_g_dc2, n_chan, 5, stride=2, pad=2, nonlinearity=lasagne.nonlinearities.sigmoid)
      l_p_mu = lasagne.layers.FlattenLayer(gen)
    elif model == 'gaussian':
      gen = Deconv2DLayer(l_g_dc2, n_chan, 5, stride=2, pad=2, nonlinearity=None)
      print lasagne.layers.get_output_shape(gen)
      l_p_mu = lasagne.layers.FlattenLayer(gen)
    gen = Deconv2DLayer(l_g_dc2, n_chan, 5, stride=2, pad=2, nonlinearity=relu_shift)
    l_p_logsigma = lasagne.layers.FlattenLayer(gen)
    l_sample = GaussianSampleLayer(l_p_mu, l_p_logsigma)

    return l_p_mu, l_p_logsigma, l_q_mu, l_q_logsigma, l_sample, l_p_z, l_d

def create_gen_objective2(X, network, deterministic=False, model='gaussian'):
    """If the gen model is a VAE"""
    # load network output
    if model == 'bernoulli':
      q_mu, q_logsigma, sample, _ \
          = lasagne.layers.get_output(network[2:-1], deterministic=deterministic)
    elif model in ('gaussian', 'svhn'):
      p_mu, p_logsigma, q_mu, q_logsigma, _, _ \
          = lasagne.layers.get_output(network[:-1], deterministic=deterministic)

    # first term of the ELBO: kl-divergence (using the closed form expression)
    kl_div = 0.5 * T.sum(1 + 2*q_logsigma - T.sqr(q_mu) 
                         - T.exp(2 * T.minimum(q_logsigma,50)), axis=1).mean()

    # second term: log-likelihood of the data under the model
    if model == 'bernoulli':
      logpxz = -lasagne.objectives.binary_crossentropy(sample, X.flatten(2)).sum(axis=1).mean()
    elif model in ('gaussian', 'svhn'):
      def log_lik(x, mu, log_sig):
          return T.sum(-(np.float32(0.5 * np.log(2 * np.pi)) + log_sig)
                        - 0.5 * T.sqr(x - mu) / T.exp(2 * log_sig), axis=1)
      logpxz = log_lik(X.flatten(2), p_mu, p_logsigma).mean()

    gen_loss = (logpxz + kl_div)

    return -gen_loss