# This file defines the links between variables in the inference and generation networks of the PoS_tagging task

import torch.nn as nn

from components.links import MLPLink, LastStateMLPLink, LSTMLink, DANLink
from sentence_classification.variables import *


def get_structured_sentence_graph(h_params, word_embeddings, pos_embeddings):

    xin_size, yin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, h_params.z_size
    xout_size, yout_size, zout_size = h_params.vocab_size, h_params.tag_size, h_params.z_size
    z_repnet = None

    # Generation
    y_gen = YvalGen(h_params, pos_embeddings)
    x_prev_gen = XPrevGen(h_params, word_embeddings, False)
    x_gen, z_gen = XGen(h_params, word_embeddings), ZGen(h_params, z_repnet, allow_prior=True)
    z_to_y = MLPLink(zin_size, h_params.pos_h, yout_size, h_params.pos_l, Categorical.parameter_activations,
                     embedding=pos_embeddings,
                     highway=h_params.highway, dropout=h_params.dropout)
    xprev_y_z_to_x = LSTMLink(xin_size+zin_size+yin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                              Categorical.parameter_activations, word_embeddings if h_params.tied_embeddings else None
                              , highway=h_params.highway, sbn=None, dropout=h_params.dropout)

    # Inference
    y_inf = YvalInfer(h_params, pos_embeddings)
    x_inf, z_inf = XInfer(h_params, word_embeddings, False), ZInfer(h_params, z_repnet)

    x_to_z = LSTMLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameter_activations,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True, last_state=True)

    x_to_y = LSTMLink(xin_size, h_params.encoder_h, yout_size, h_params.encoder_l, Categorical.parameter_activations,
                      #embedding=pos_embeddings,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True, last_state=True)
    x_to_y.rnn = x_to_z.rnn


    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_y, y_inf]),
                                    nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([z_gen, z_to_y, y_gen]),
                                    nn.ModuleList([y_gen, xprev_y_z_to_x, x_gen]),
                                    nn.ModuleList([z_gen, xprev_y_z_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_y_z_to_x, x_gen])
                                    ])}, y_inf, x_gen


def get_zy_sentence_graph(h_params, word_embeddings, pos_embeddings):

    xin_size, yin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, h_params.z_size
    xout_size, yout_size, zout_size = h_params.vocab_size, h_params.tag_size, h_params.z_size
    z_repnet = None

    # Generation
    y_gen = YvalGen(h_params, pos_embeddings, allow_prior=True)
    x_prev_gen = XPrevGen(h_params, word_embeddings, False)
    x_gen, z_gen = XGen(h_params, word_embeddings), ZGen(h_params, z_repnet, allow_prior=True)
    xprev_y_z_to_x = LSTMLink(xin_size+zin_size+yin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                              Categorical.parameter_activations, word_embeddings if h_params.tied_embeddings else None
                              , highway=h_params.highway, sbn=None, dropout=h_params.dropout)

    # Inference
    y_inf = YvalInfer(h_params, pos_embeddings)
    x_inf, z_inf = XInfer(h_params, word_embeddings, False), ZInfer(h_params, z_repnet)

    x_to_z = LSTMLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameter_activations,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True, last_state=True)

    x_to_y = LSTMLink(xin_size, h_params.encoder_h, yout_size, h_params.encoder_l, Categorical.parameter_activations,
                      #embedding=pos_embeddings,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True, last_state=True)
    x_to_y.rnn = x_to_z.rnn

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_y, y_inf]),
                                    nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([y_gen, xprev_y_z_to_x, x_gen]),
                                    nn.ModuleList([z_gen, xprev_y_z_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_y_z_to_x, x_gen])
                                    ])}, y_inf, x_gen


def get_y_sentence_graph(h_params, word_embeddings, pos_embeddings):

    xin_size, yin_size = h_params.embedding_dim, h_params.pos_embedding_dim
    xout_size, yout_size = h_params.vocab_size, h_params.tag_size

    # Generation
    y_gen = YvalGen(h_params, pos_embeddings, allow_prior=True)
    x_prev_gen = XPrevGen(h_params, word_embeddings, False)
    x_gen = XGen(h_params, word_embeddings)
    xprev_y_to_x = LSTMLink(xin_size+yin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                              Categorical.parameter_activations, word_embeddings if h_params.tied_embeddings else None
                              , highway=h_params.highway, sbn=None, dropout=h_params.dropout)

    # Inference
    y_inf = YvalInfer(h_params, pos_embeddings)
    x_inf = XInfer(h_params, word_embeddings, False)

    x_to_y = LSTMLink(xin_size, h_params.encoder_h, yout_size, h_params.encoder_l, Categorical.parameter_activations,
                      #embedding=pos_embeddings,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True, last_state=True)
    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_y, y_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([y_gen, xprev_y_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_y_to_x, x_gen])
                                    ])}, y_inf, x_gen


def get_structured_zy_token_graph(h_params, word_embeddings, pos_embeddings):

    xin_size, yin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, h_params.z_size
    xout_size, yout_size, zout_size = h_params.vocab_size, h_params.tag_size, h_params.z_size
    z_repnet = None

    # Generation
    y_gen = YvalGen(h_params, pos_embeddings, sequence_lv=False)
    x_prev_gen = XPrevGen(h_params, word_embeddings, False)
    x_gen, z_gen = XGen(h_params, word_embeddings), ZGen(h_params, z_repnet, allow_prior=True, sequence_lv=False)
    xprev_to_z = LSTMLink(xin_size, h_params.decoder_h, zout_size, h_params.decoder_l,
                          Gaussian.parameter_activations, highway=h_params.highway, sbn=None,
                          dropout=h_params.dropout)
    z_to_y = MLPLink(zin_size, h_params.pos_h, yout_size, h_params.pos_l, Categorical.parameter_activations,
                     embedding=pos_embeddings, highway=h_params.highway, dropout=h_params.dropout)
    xprev_y_z_to_x = LSTMLink(xin_size+zin_size+yin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                              Categorical.parameter_activations, word_embeddings if h_params.tied_embeddings else None,
                              highway=h_params.highway, sbn=None, dropout=h_params.dropout)

    # Inference
    y_inf = YvalInfer(h_params, pos_embeddings, sequence_lv=False)
    x_inf, z_inf = XInfer(h_params, word_embeddings, False), ZInfer(h_params, z_repnet, sequence_lv=False)

    x_to_z = LSTMLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameter_activations,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True)

    x_to_y = LSTMLink(xin_size, h_params.encoder_h, yout_size, h_params.encoder_l, Categorical.parameter_activations,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True)
    x_to_y.rnn = x_to_z.rnn

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_y, y_inf]),
                                    nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([z_gen, z_to_y, y_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_to_z, z_gen]),
                                    nn.ModuleList([y_gen, xprev_y_z_to_x, x_gen]),
                                    nn.ModuleList([z_gen, xprev_y_z_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_y_z_to_x, x_gen])
                                    ])}, y_inf, x_gen


def get_zy_token_graph(h_params, word_embeddings, pos_embeddings):
    xin_size, yin_size, zin_size = h_params.embedding_dim, h_params.pos_embedding_dim, h_params.z_size
    xout_size, yout_size, zout_size = h_params.vocab_size, h_params.tag_size, h_params.z_size
    z_repnet = None

    # Generation
    y_gen = YvalGen(h_params, pos_embeddings, sequence_lv=False, allow_prior=True)
    x_prev_gen = XPrevGen(h_params, word_embeddings, False)
    x_gen, z_gen = XGen(h_params, word_embeddings), ZGen(h_params, z_repnet, allow_prior=True, sequence_lv=False)

    xprev_y_z_to_x = LSTMLink(xin_size+zin_size+yin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                              Categorical.parameter_activations, word_embeddings if h_params.tied_embeddings else None,
                              highway=h_params.highway, sbn=None, dropout=h_params.dropout)

    # Inference
    y_inf = YvalInfer(h_params, pos_embeddings, sequence_lv=False)
    x_inf, z_inf = XInfer(h_params, word_embeddings, False), ZInfer(h_params, z_repnet, sequence_lv=False)

    x_to_z = LSTMLink(xin_size, h_params.encoder_h, zout_size, h_params.encoder_l, Gaussian.parameter_activations,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True)

    x_to_y = LSTMLink(xin_size, h_params.encoder_h, yout_size, h_params.encoder_l, Categorical.parameter_activations,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True)
    x_to_y.rnn = x_to_z.rnn

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_y, y_inf]),
                                    nn.ModuleList([x_inf, x_to_z, z_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([y_gen, xprev_y_z_to_x, x_gen]),
                                    nn.ModuleList([z_gen, xprev_y_z_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_y_z_to_x, x_gen])
                                    ])}, y_inf, x_gen


def get_y_token_graph(h_params, word_embeddings, pos_embeddings):
    xin_size, yin_size = h_params.embedding_dim, h_params.pos_embedding_dim
    xout_size, yout_size = h_params.vocab_size, h_params.tag_size

    # Generation
    y_gen = YvalGen(h_params, pos_embeddings, sequence_lv=False, allow_prior=True)
    x_prev_gen = XPrevGen(h_params, word_embeddings, False)
    x_gen = XGen(h_params, word_embeddings)

    xprev_y_to_x = LSTMLink(xin_size+yin_size, h_params.decoder_h, xout_size, h_params.decoder_l,
                            Categorical.parameter_activations, word_embeddings if h_params.tied_embeddings else None,
                            highway=h_params.highway, sbn=None, dropout=h_params.dropout)

    # Inference
    y_inf = YvalInfer(h_params, pos_embeddings, sequence_lv=False)
    x_inf = XInfer(h_params, word_embeddings, False)

    x_to_y = LSTMLink(xin_size, h_params.encoder_h, yout_size, h_params.encoder_l, Categorical.parameter_activations,
                      highway=h_params.highway, dropout=h_params.dropout, bidirectional=True)

    return {'infer': nn.ModuleList([nn.ModuleList([x_inf, x_to_y, y_inf]),
                                    ]),
            'gen':   nn.ModuleList([nn.ModuleList([y_gen, xprev_y_to_x, x_gen]),
                                    nn.ModuleList([x_prev_gen, xprev_y_to_x, x_gen])
                                    ])}, y_inf, x_gen
