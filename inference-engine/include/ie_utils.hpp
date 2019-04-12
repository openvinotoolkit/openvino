// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file that provides utilities for calculating per layer theoretical statistic
 * @file ie_utils.hpp
 */
#pragma once
#include <unordered_map>
#include <string>
#include <cpp/ie_cnn_network.h>

namespace InferenceEngine {

/**
 * @brief Contains information about floating point operations
 * and common size of parameter blobs.
 */
struct LayerComplexity {
    /** @brief Number of floating point operations for reference implementation */
    size_t flops;
    /** @brief Total size of parameter blobs */
    size_t params;
};

/**
 * @brief Computes per layer theoretical computational and memory
 * complexity.
 *
 * @param network input graph
 * @return map from layer name to layer complexity
 */
std::unordered_map<std::string, LayerComplexity> getNetworkComplexity(const InferenceEngine::ICNNNetwork &network);

}  // namespace InferenceEngine

