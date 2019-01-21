// Copyright (C) 2018 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief a header file for internal Layers structure
 * @file
 */
#pragma once

#include "ie_layers.h"
#include <string>

namespace InferenceEngine {

/**
 * LSTM Cell Layer
 *
 * Inputs:
 *    Xt   {N, D}
 *    Ht-1 {N, S}
 *    Ct-1 {N, S}
 *
 * Outputs:
 *    Ht {N, S}
 *    Ct {N, S}
 *
 * Weights:
 *    W {G=4, S, D+S}
 *    B {G=4, S}
 *
 * G=4 and gate order is [f,i,c,o]
 *
 * Semantic:
 *
 *   *  - matrix mult
 *  (.) - eltwise mult
 *  [,] - concatenation
 *
 *  f = sigmoid
 *  h = tanh
 *
 * - ft = f(Wf*[Ht-1, Xt] + Bf)
 * - it = f(Wi*[Ht-1, Xt] + Bi)
 * - ct = h(Wc*[Ht-1, Xt] + Bc)
 * - ot = f(Wo*[Ht-1, Xt] + Bo)
 * - Ct = ft (.) Ct-1 + it (.) ct
 * - Ht = ot (.) h(Ct)
 */
class LSTMCell : public WeightableLayer {
public:
    using WeightableLayer::WeightableLayer;
};

/**
 * @brief This class represents RNN-Sequence layer
 *
 * Date shapes and meaning (cellType = "LSTM", axis = 1):
 *   input[0] Xt - {N,T,DC} input data sequence
 *   input[1] H0 - {N,SC}   initial hidden state
 *   input[2] C0 - {N,SC}   initial cell state
 *
 *   output[0] Ht - {N,T,SC} out data sequence
 *   output[1] HT - {N,SC}   last hidden state
 *   output[2] CT - {N,SC}   last cell state
 *
 *   Recurrent formula and weight format are same as from
 *   corresponding Cell primitive.
 */
class RNNLayer : public WeightableLayer {
public:
    /**
     * @brief Type of RNN cell used sequence layer
     * Possible values "RNN", "LSTM", "GRU".
     */
    std::string cellType = "LSTM";

    /**
     * @brief An axis by which iteration is performed
     * axis=0 means first input/output data blob dimension is sequence
     * axis=1 means first input/output data blob dimension is batch
     */
    unsigned int axis = 1;

    /**
     * @brief Direction of iteration through sequence dimension
     */
    enum Direction {
        RNN_FWD,  /**< Forward mode. Iterate starts from index 0 with step 1.         */
        RNN_BWD,  /**< Backward mode. Iterate starts from last index with step -1.    */
        RNN_BDR   /**< Bidirectional mode. First is forward pass, second is backward. */
    };

    Direction direction = RNN_FWD;

    using WeightableLayer::WeightableLayer;
};

}  // namespace InferenceEngine
