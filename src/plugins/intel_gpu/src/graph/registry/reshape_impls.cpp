// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "primitive_inst.h"

#ifdef OV_GPU_WITH_VULKAN_RT
    #include "impls/vulkan/reshape.hpp"
#endif

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<reshape>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
#ifdef OV_GPU_WITH_VULKAN_RT
        std::make_shared<vulkan::ReshapeImplementationManager>(shape_types::static_shape),
#endif
        OV_GPU_GET_INSTANCE_OCL(reshape, shape_types::static_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
