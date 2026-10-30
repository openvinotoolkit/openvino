// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/gated_residual.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_OCL
#    include "impls/ocl_v2/gated_residual/gated_residual.hpp"
#endif
#if OV_GPU_WITH_CM
#    include "impls/cm/gated_residual/gated_residual.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<gated_residual>::get_implementations() {
    // OpenCL is the default; the CM implementation is only used with OV_GPU_GR_IMPL=cm.
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#if OV_GPU_WITH_OCL
        OV_GPU_CREATE_INSTANCE_OCL(ocl::GatedResidualImplementationManager, shape_types::any)
#endif
#if OV_GPU_WITH_CM
            OV_GPU_CREATE_INSTANCE_CM(cm::GatedResidualCMImplementationManager, shape_types::any)
#endif
    };

    return impls;
}

}  // namespace ov::intel_gpu
