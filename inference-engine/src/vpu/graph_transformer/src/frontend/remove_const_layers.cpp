// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/compile_env.hpp"
#include "graph_transformer.h"

#include "cnn_network_impl.hpp"
#include "cnn_network_ngraph_impl.hpp"

namespace vpu {

void FrontEnd::removeConstLayers(ie::ICNNNetwork& network) {
    VPU_PROFILE(removeConstLayers);

    const auto& env = CompileEnv::get();

    env.log->trace("Remove const layers");
    VPU_LOGGER_SECTION(env.log);

    ie::ICNNNetwork* cnnNetwork = &network;
    if (auto nGraphImpl = dynamic_cast<ie::details::CNNNetworkNGraphImpl*>(&network)) {
        // NGraph implementation cannot be casted to CNNNetworkImpl directly
        cnnNetwork = nGraphImpl->getCNNNetwork().get();
    }

    // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
    if (auto cnnNetworkImpl = dynamic_cast<ie::details::CNNNetworkImpl*>(cnnNetwork)) {
        ie::ConstTransformer(cnnNetworkImpl).fullTrim();
    }
}

}  // namespace vpu
