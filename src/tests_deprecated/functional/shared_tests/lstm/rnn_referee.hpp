// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_blob.h"
#include "rnn_util.hpp"
#include <vector>

enum Cell {
    LSTM,    /**< Vanilla LSTMCell */
    GRU,     /**< Vanilla GRUCell */
    RNN,     /**< Vanilla RNNCell */
    GRU_lbr  /**< Vanilla GRUCell */
};

/**
 * Descriptor of activation function
 * type : [sigm, tanh, relu, ...]
 * alpha, beta : optional
 */
struct ActivationDesc {
    std::string alg;
    float alpha;
    float beta;
};
using ActivationSet = std::vector<ActivationDesc>;

/**
 * Descriptor of general RNN cell
 */
struct CellDesc {
    Cell type;                 /**< Type of RNN cell */
    ActivationSet acts;        /**< Activations aplgorithm */
    float clip;                /**< Clip value. 0 - no clipping */
};

/**
 * Ref scoring for some RNN cells
 * Provide weight filler and in_data filler and out_data checker
 */
class RNN_Referee {
public:
    static std::shared_ptr<RNN_Referee> create_referee(CellDesc cell, size_t N, size_t T, size_t D, size_t S);
    virtual ~RNN_Referee() = default;

    virtual void wFiller(InferenceEngine::Blob::Ptr) = 0;
    virtual void bFiller(InferenceEngine::Blob::Ptr) = 0;

    virtual size_t wSize() = 0;
    virtual size_t bSize() = 0;

    virtual size_t stateNum() = 0;

    virtual const std::vector<Filler>& getDataFillers() = 0;
    virtual const std::vector<Checker>& getDataChecker() = 0;
};
