// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <cpp/ie_cnn_network.h>

namespace vpu {

InferenceEngine::QueryNetworkResult getQueryNetwork(const InferenceEngine::CNNNetwork& convertedNetwork,
                                                    const std::shared_ptr<const ngraph::Function>& function,
                                                    const std::string& pluginName, const std::set<std::string>& supportedLayers);

} // namespace vpu
