// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/paged_selective_ssm.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_OCL
#    include "impls/ocl_v2/paged_selective_ssm.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<paged_selective_ssm>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#if OV_GPU_WITH_OCL
        OV_GPU_CREATE_INSTANCE_OCL(ocl::PagedSelectiveSSMJitIntegrated, shape_types::any)
            OV_GPU_CREATE_INSTANCE_OCL(ocl::PagedSelectiveSSMJitDiscrete, shape_types::any)
                OV_GPU_CREATE_INSTANCE_OCL(ocl::PagedSelectiveSSMOpt, shape_types::any)
#endif
    };

    return impls;
}

}  // namespace ov::intel_gpu
