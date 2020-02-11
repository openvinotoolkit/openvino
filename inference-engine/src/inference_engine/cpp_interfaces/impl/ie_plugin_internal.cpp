// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ie_plugin_internal.hpp"
#ifdef ENABLE_NGRAPH
#include "cnn_network_ngraph_impl.hpp"
#endif
#include <memory>

std::shared_ptr<InferenceEngine::ICNNNetwork> InferenceEngine::InferencePluginInternal::ConvertAndCloneNetwork(ICNNNetwork& network) {
#ifdef ENABLE_NGRAPH
    if (auto networkNGraph = dynamic_cast<CNNNetworkNGraphImpl*>(&network)) {
        auto nGraphNetwork = networkNGraph->cloneNGraphImpl();
        nGraphNet = nGraphNetwork;
        return nGraphNetwork->getCNNNetwork();
    }
#endif
    return CloneNetwork(network);
}
