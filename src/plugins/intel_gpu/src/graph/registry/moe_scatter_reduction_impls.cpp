// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/moe_scatter_reduction.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/moe/moe_scatter_reduction_ref.hpp"
    #include "impls/ocl_v2/moe/moe_scatter_reduction_opt.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<moe_scatter_reduction>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::MoeScatterReductionOpt, shape_types::dynamic_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::MoeScatterReductionOpt, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::MoeScatterReductionRef, shape_types::dynamic_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::MoeScatterReductionRef, shape_types::static_shape)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov

