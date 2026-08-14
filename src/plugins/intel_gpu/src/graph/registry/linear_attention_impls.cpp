// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/linear_attention.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_CM
    #include "impls/cm/linear_attention_opt.hpp"
#endif
#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/linear_attention_ref.hpp"
#endif
namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<linear_attention>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
//#if OV_GPU_WITH_CM
//        OV_GPU_CREATE_INSTANCE_CM(cm::LinearAttentionOptImplementationManager, shape_types::any)
//#endif
#if OV_GPU_WITH_OCL
    OV_GPU_CREATE_INSTANCE_OCL(ocl::LinearAttentionRef, shape_types::any)
#endif
    };

    return impls;
}

} // namespace ov::intel_gpu
