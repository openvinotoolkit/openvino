// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/compile_env.hpp"
#include "graph_transformer.h"

#include "cnn_network_impl.hpp"

namespace vpu {

void FrontEnd::removeConstLayers(ie::ICNNNetwork& network) {
    VPU_PROFILE(removeConstLayers);

    const auto& env = CompileEnv::get();

    env.log->trace("Remove const layers");
    VPU_LOGGER_SECTION(env.log);

    auto implNetwork = dynamic_cast<ie::details::CNNNetworkImpl *>(&network);
    VPU_THROW_UNLESS(implNetwork != nullptr, "FrontEnd::removeConstLayers expects CNNNetworkImpl");

    ie::ConstTransformer(implNetwork).fullTrim();
}

}  // namespace vpu
