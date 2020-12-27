// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <ngraph/ngraph.hpp>
#include <ie_icnn_network.hpp>

namespace vpu {

InferenceEngine::QueryNetworkResult getQueryNetwork(const InferenceEngine::ICNNNetwork::Ptr& convertedNetwork,
                                                    const std::shared_ptr<const ngraph::Function>& function,
                                                    const std::string& pluginName, const std::set<std::string>& supportedLayers);

} // namespace vpu
