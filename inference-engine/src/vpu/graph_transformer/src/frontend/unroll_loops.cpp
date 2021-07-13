// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <vpu/frontend/frontend.hpp>

#include <string>

#include <legacy/net_pass.h>
#include <legacy/details/ie_cnn_network_iterator.hpp>

#include <vpu/compile_env.hpp>

#include <vpu/configuration/options/ir_with_scales_directory.hpp>
#include <vpu/configuration/options/force_pure_tensor_iterator.hpp>
#include <vpu/configuration/options/enable_tensor_iterator_unrolling.hpp>

namespace vpu {

void FrontEnd::unrollLoops(ie::CNNNetwork& network) {
    VPU_PROFILE(unrollLoops);

    const auto& env = CompileEnv::get();

    env.log->trace("Unroll TensorIterator loops");
    VPU_LOGGER_SECTION(env.log);

    if (!env.config.get<IRWithScalesDirectoryOption>().empty()) {
        // TODO: Scale dumps does not work with IR, which contain Tensor Iterator layers, because we cannot serialize them. #-23429
        for (auto iterator = ie::details::CNNNetworkIterator(network); iterator != ie::details::CNNNetworkIterator(); ++iterator) {
            const auto& layer = *iterator;
            VPU_THROW_UNLESS(!ie::details::CaselessEq<std::string>()(layer->type, "TensorIterator"),
                "Scale dumps does not work with IR, which contain Tensor Iterator layers.");
        }
    }

    if (env.config.get<ForcePureTensorIteratorOption>()) {
        return;
    }

    if (env.config.get<EnableTensorIteratorUnrollingOption>()) {
        ie::NetPass::UnrollTI(network);
    } else {
        // Try to convert network to a RNN sequence due to performance reasons
        ie::NetPass::CombineRNNSeq(network);
    }
}

}  // namespace vpu
