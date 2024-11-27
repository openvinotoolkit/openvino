// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/gather_nd.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/gather_nd.hpp"
#endif


namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<gather_nd>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::GatherNDImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::GatherNDImplementationManager, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
