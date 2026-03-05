// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/fused_conv.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/fused_conv_ref.hpp"
#endif
namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<fused_conv>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#if OV_GPU_WITH_OCL
    OV_GPU_CREATE_INSTANCE_OCL(ocl::FusedConvRef, shape_types::any)
#endif
    };

    return impls;
}

} // namespace ov::intel_gpu
