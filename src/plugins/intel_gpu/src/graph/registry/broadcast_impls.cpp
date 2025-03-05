// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "predicates.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/broadcast.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<broadcast>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_OCL(broadcast, shape_types::static_shape, not_in_shape_flow())
        OV_GPU_GET_INSTANCE_OCL(broadcast, shape_types::dynamic_shape, not_in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(broadcast, shape_types::static_shape, in_shape_flow())
        OV_GPU_GET_INSTANCE_CPU(broadcast, shape_types::dynamic_shape, in_shape_flow())
    };

    return impls;
}

}  // namespace ov::intel_gpu
