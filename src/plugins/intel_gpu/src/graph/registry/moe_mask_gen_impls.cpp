// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/moe_mask_gen.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<moe_mask_gen>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_CPU(moe_mask_gen, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_CPU(moe_mask_gen, shape_types::dynamic_shape)
    };

    return impls;
}

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<moe_mask_gen_reshape>::get_implementations() {
    // This primitive only used for shape inference, so no implelementation
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_CPU(moe_mask_gen_reshape, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_CPU(moe_mask_gen_reshape, shape_types::dynamic_shape)
    };

    return impls;
}
}  // namespace ov::intel_gpu
