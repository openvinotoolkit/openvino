// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

/**
 * @brief A header file for CNNNetwork tools
 * 
 * @file ie_cnn_network_tools.h
 */
#pragma once
#include <vector>

#include "cpp/ie_cnn_network.h"

#include <legacy/ie_layers.h>

namespace InferenceEngine {
namespace details {

std::vector<CNNLayerPtr> CNNNetSortTopologically(const CNNNetwork& network);

}  // namespace details
}  // namespace InferenceEngine
