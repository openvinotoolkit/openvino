// Copyright (C) 2018 Intel Corporation
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
 * @param net network to modify
 * @return true if all Tensor iterator was converted
 */
INFERENCE_ENGINE_API_CPP(bool) CombineLSTMSeq(const ICNNNetwork &net);

/**
 * Unroll all present Tensor Iterators
 * @param net network to modify
 * @return true if all Tensor iterator was unrolled successfully
 */
INFERENCE_ENGINE_API_CPP(bool) UnrollTI(const ICNNNetwork &net);

}  // namespace NetPass
}  // namespace InferenceEngine
