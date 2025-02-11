// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/scaled_dot_product_attention.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl_new/sdpa_ref.hpp"
    #include "impls/ocl_new/sdpa_opt.hpp"
#endif

#if OV_GPU_WITH_OCL && OV_GPU_WITH_ONEDNN
    #include "impls/ocl_new/sdpa_micro.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<scaled_dot_product_attention>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#if OV_GPU_WITH_ONEDNN
        OV_GPU_CREATE_INSTANCE_OCL(ocl::SDPAMicro, shape_types::any)
#endif
        OV_GPU_CREATE_INSTANCE_OCL(ocl::SDPAOpt, shape_types::any)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::SDPARef, shape_types::any)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
