// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ie_icnn_network.hpp>
#include <legacy/ie_layers.h>

#include <string>
#include <vector>

namespace InferenceEngine {
namespace Serialization {

/**
    * @brief Serialize execution network into IE IR-like XML file
    * @param xmlPath   Path to XML file
    * @param network   network to be serialized
    */
INFERENCE_ENGINE_API_CPP(void) Serialize(const std::string& xmlPath, const InferenceEngine::ICNNNetwork& network);

/**
    * @brief Returns set of topologically sorted layers
    * @param network network to be sorted
    * @return `std::vector` of topologically sorted CNN layers
    */
INFERENCE_ENGINE_API_CPP(std::vector<CNNLayerPtr>) TopologicalSort(const InferenceEngine::ICNNNetwork& network);

}  // namespace Serialization
}  // namespace InferenceEngine
