// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/paged_attention.hpp"
#include "registry.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/sdpa/paged_attention_opt.hpp"
#endif

#if OV_GPU_WITH_CM
    #include "impls/cm/paged_attention.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<paged_attention>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::PagedAttentionOpt, shape_types::any)
        OV_GPU_CREATE_INSTANCE_CM(cm::PagedAttentionImplementationManager, shape_types::any)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
