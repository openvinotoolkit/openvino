// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/moe_gather.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<moe_gather>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_CPU(moe_gather, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_CPU(moe_gather, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
