// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ie_icnn_network.hpp"

#include <vector>
#include <string>
#include <map>

namespace InferenceEngine {
namespace NetPass {

/**
 * Try to detect LSTM Sequence pattern inside TI and convert it
 *
 * @param net network to modify
 * @return true if all Tensor iterator was converted
 */
INFERENCE_ENGINE_API_CPP(bool) CombineRNNSeq(ICNNNetwork &net);
INFERENCE_ENGINE_API_CPP(bool) CombineRNNSeq(TensorIterator::Body &net);

/**
 * Returns a vector of the topologically sorted layers from
 * the passed TI layer body.
 *
 * @param body TI body
 * @return vector of layer objects
 */
INFERENCE_ENGINE_API_CPP(std::vector<CNNLayerPtr>) TIBodySortTopologically(const TensorIterator::Body &body);

/**
 * Returns a vector of the topologically sorted layers from
 * the passed TI layer body.
 *
 * @param body TI body
 * @return vector of layer objects
 */
TensorIterator::Body CopyTIBody(const TensorIterator::Body &body, std::string suffix = std::string());

/**
 * Unroll all present Tensor Iterators
 *
 * @param net network to modify
 * @return true if all Tensor iterator was unrolled successfully
 */
INFERENCE_ENGINE_API_CPP(bool) UnrollTI(ICNNNetwork &net);

/**
 * Unroll all RNN specific layers by predicate
 *
 * Will be applied to all RNNSeq and RNNCell layers
 *
 * @param net network to modify
 * @param pred predicate to mark layer to unroll
 * @return true if all RNN layers was unrolled successfully
 */
INFERENCE_ENGINE_API_CPP(bool) UnrollRNN_if(ICNNNetwork &net,
        std::function<bool(const RNNCellBase&)> pred);

INFERENCE_ENGINE_API_CPP(bool) UnrollRNN_if(TensorIterator::Body &net,
        std::function<bool(const RNNCellBase&)> pred);


}  // namespace NetPass
}  // namespace InferenceEngine
