// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <cnn_network_impl.hpp>
#include <graph_transformer.h>

#include <vpu/compile_env.hpp>

namespace vpu {

void FrontEnd::RemoveConstLayers(ie::ICNNNetwork& network) {
    VPU_PROFILE(RemoveConstLayers);

    const auto& env = CompileEnv::get();

    env.log->debug("Remove const layers");
    VPU_LOGGER_SECTION(env.log);

    // valid for CNNNetworkImpl only, while there's no API in ICNNNetwork to change network
    if (auto* implNetwork = dynamic_cast<ie::details::CNNNetworkImpl*>(&network)) {
        ie::ConstTransformer transformator(implNetwork);
        transformator.fullTrim();
    }
}

}  // namespace vpu
