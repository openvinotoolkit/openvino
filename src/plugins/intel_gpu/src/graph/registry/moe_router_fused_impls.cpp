// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/moe_router_fused.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_OCL
#    include "impls/ocl_v2/moe/moe_router_fused_opt.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<moe_router_fused>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {OV_GPU_CREATE_INSTANCE_OCL(ocl::MoeRouterFusedOpt, shape_types::any)};

    return impls;
}

}  // namespace ov::intel_gpu
