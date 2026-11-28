// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/gemm.hpp"
#include "intel_gpu/runtime/device.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/gemm_onednn.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<gemm>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::GemmImplementationManager, shape_types::static_shape)
        // Intel GPUs use a reference kernel for oneDNN matmul with runtime dimensions.
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::GemmImplementationManager, shape_types::dynamic_shape,
            [](const program_node& node) {
                return node.get_program().get_engine().get_device_info().vendor_id != INTEL_VENDOR_ID;
        })
        OV_GPU_GET_INSTANCE_OCL(gemm, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(gemm, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
