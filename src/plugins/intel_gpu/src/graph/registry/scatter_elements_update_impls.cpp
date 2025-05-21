// Copyright (C) 2024-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "predicates.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/scatter_elements_update.hpp"
#include "primitive_inst.h"

#if OV_GPU_WITH_OCL
    #include "impls/ocl/scatter_elements_update.hpp"
#endif


namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<scatter_elements_update>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ScatterElementsUpdateImplementationManager, shape_types::static_shape)
        OV_GPU_CREATE_INSTANCE_OCL(ocl::ScatterElementsUpdateImplementationManager, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
