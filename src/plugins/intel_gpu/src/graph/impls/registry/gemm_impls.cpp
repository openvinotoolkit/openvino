// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/gemm.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/gemm_onednn.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<gemm>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::GemmImplementationManager, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(gemm, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(gemm, shape_types::dynamic_shape,
            [](const program_node& node) {
                return !node.can_use(impl_types::onednn);
        })
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
