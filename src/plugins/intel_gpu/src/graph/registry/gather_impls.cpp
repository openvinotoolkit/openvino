// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "predicates.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/gather.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/gather.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<gather>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::GatherImplementationManager, shape_types::static_shape, not_in_shape_flow())
        OV_GPU_CREATE_INSTANCE_OCL(ocl::GatherImplementationManager, shape_types::dynamic_shape, not_in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(gather, shape_types::static_shape, in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(gather, shape_types::dynamic_shape, in_shape_flow())
    };

    return impls;
}

}  // namespace ov::intel_gpu
