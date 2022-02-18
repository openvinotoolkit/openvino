// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <memory>

#include "cnn_network_ngraph_impl.hpp"

namespace InferenceEngine {
namespace details {
/**
 * @brief Checks if the input network is batch-able (e.g. no dynamic inputs, inputs has the batch dimension, etc)
 * @param function A ngraph function to check for automatic-batching applicability
 * @return A boolean value indicating whether the network can be safely batched
 */
enum NetworkBatchAbility : uint32_t { NO = 0, AS_IS, WITH_HETERO };
NetworkBatchAbility isNetworkBatchable(const CNNNetwork& network, const std::string& deviceNoBatch);

}  // namespace details
}  // namespace InferenceEngine
