// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/frontend/frontend.hpp"
#include "vpu/compile_env.hpp"
#include <legacy/graph_transformer.h>

#include <legacy/cnn_network_impl.hpp>

namespace vpu {

void FrontEnd::removeConstLayers(ie::CNNNetwork& network) {
    VPU_PROFILE(removeConstLayers);

    const auto& env = CompileEnv::get();

    env.log->trace("Remove const layers");
    VPU_LOGGER_SECTION(env.log);

    IE_SUPPRESS_DEPRECATED_START
    auto & icnnnet = static_cast<ie::ICNNNetwork &>(network);
    auto implNetwork = dynamic_cast<ie::details::CNNNetworkImpl *>(&icnnnet);
    VPU_THROW_UNLESS(implNetwork != nullptr, "FrontEnd::removeConstLayers expects CNNNetworkImpl");

    ie::ConstTransformer(implNetwork).fullTrim();
    IE_SUPPRESS_DEPRECATED_END
}

}  // namespace vpu
