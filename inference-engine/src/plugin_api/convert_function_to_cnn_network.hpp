// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cnn_network_ngraph_impl.hpp"

#include <memory>

namespace InferenceEngine {
namespace details {
INFERENCE_ENGINE_API_CPP(std::shared_ptr<CNNNetworkImpl>)
convertFunctionToICNNNetwork(const std::shared_ptr<const ::ngraph::Function>& graph, const CNNNetworkNGraphImpl &nGraphImpl);
}  // namespace details
}  // namespace InferenceEngine
