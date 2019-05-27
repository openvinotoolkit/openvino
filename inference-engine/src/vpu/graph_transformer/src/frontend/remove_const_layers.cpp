// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>
#include <cnn_network_impl.hpp>
#include <graph_transformer.h>

namespace vpu {

void FrontEnd::RemoveConstLayers(ie::ICNNNetwork &network) {
    VPU_PROFILE(RemoveConstLayers);
    auto *implNetwork = dynamic_cast<InferenceEngine::details::CNNNetworkImpl *>(&network);
    if (implNetwork) {
        // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
        InferenceEngine::ConstTransformer transformator(implNetwork);
        transformator.fullTrim();
    }
}

}  // namespace vpu