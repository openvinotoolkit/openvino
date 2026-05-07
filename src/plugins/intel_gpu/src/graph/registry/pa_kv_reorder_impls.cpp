// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/pa_kv_reorder.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_OCL
#    include "impls/ocl_v2/pa_kv_reorder.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<pa_kv_reorder>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {OV_GPU_CREATE_INSTANCE_OCL(ocl::PA_KV_reorder, shape_types::any)};

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
