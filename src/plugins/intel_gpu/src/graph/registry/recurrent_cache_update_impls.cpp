// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/primitives/recurrent_cache_update.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

#if OV_GPU_WITH_OCL
#    include "impls/ocl_v2/recurrent_cache_update.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<recurrent_cache_update>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#if OV_GPU_WITH_OCL
        OV_GPU_CREATE_INSTANCE_OCL(ocl::RecurrentCacheUpdate, shape_types::any)
#endif
    };

    return impls;
}

}  // namespace ov::intel_gpu
