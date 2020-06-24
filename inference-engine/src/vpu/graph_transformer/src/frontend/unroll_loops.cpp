// Copyright (C) 2018-2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>

#include <net_pass.h>
#include <details/ie_cnn_network_iterator.hpp>
#include <details/caseless.hpp>

#include <vpu/compile_env.hpp>

namespace vpu {

void FrontEnd::unrollLoops(ie::ICNNNetwork& network) {
    VPU_PROFILE(unrollLoops);

    const auto& env = CompileEnv::get();

    env.log->trace("Unroll TensorIterator loops");
    VPU_LOGGER_SECTION(env.log);

    if (!env.config.irWithVpuScalesDir.empty()) {
        // TODO: Scale dumps does not work with IR, which contain Tensor Iterator layers, because we cannot serialize them. #-23429
        for (auto iterator = ie::details::CNNNetworkIterator(&network); iterator != ie::details::CNNNetworkIterator(); ++iterator) {
            const auto& layer = *iterator;
            VPU_THROW_UNLESS(!ie::details::CaselessEq<std::string>()(layer->type, "TensorIterator"),
                "Scale dumps does not work with IR, which contain Tensor Iterator layers.");
        }
    }

    if (env.config.forcePureTensorIterator) {
        return;
    }

    if (env.config.enableTensorIteratorUnrolling) {
        ie::NetPass::UnrollTI(network);
    } else {
        // Try to convert network to a RNN sequence due to performance reasons
        ie::NetPass::CombineRNNSeq(network);
    }
}

}  // namespace vpu
