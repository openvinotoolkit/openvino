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

}  // namespace NetPass
}  // namespace InferenceEngine
