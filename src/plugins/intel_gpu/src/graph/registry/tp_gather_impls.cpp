// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/tp_gather.hpp"
#include "primitive_inst.h"

#ifdef ENABLE_TP

#if OV_GPU_WITH_OCL
    #include "impls/ocl_v2/tp/tp_gather.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<tp_gather>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::TPGather, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::TPGather, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu

#endif  // ENABLE_TP
