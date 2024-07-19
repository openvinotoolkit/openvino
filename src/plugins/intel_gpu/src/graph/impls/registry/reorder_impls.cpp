// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/reorder.hpp"

#if OV_GPU_WITH_ONEDNN
    #include "impls/onednn/reorder_onednn.hpp"
#endif
#if OV_GPU_WITH_OCL
    #include "impls/ocl/reorder.hpp"
#endif

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<reorder>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_ONEDNN(onednn::ReorderImplementationManager, shape_types::static_shape),
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ReorderImplementationManager, shape_types::static_shape),
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ReorderImplementationManager, shape_types::dynamic_shape),
        OV_GPU_GET_INSTANCE_CPU(reorder, shape_types::static_shape),
        OV_GPU_GET_INSTANCE_CPU(reorder, shape_types::dynamic_shape),
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
