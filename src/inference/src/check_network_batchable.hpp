// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "ngraph/function.hpp"

namespace InferenceEngine {
namespace details {
/**
 * @brief Checks if the input network is batch-able (e.g. no dynamic inputs, inputs has the batch dimension, etc)
 * @param function A ngraph function to check for automatic-batching applicability
 * @return A boolean value indicating whether the network can be safely batched
 */
bool isNetworkBatchable(std::shared_ptr<ngraph::Function> function);

}  // namespace details
}  // namespace InferenceEngine
