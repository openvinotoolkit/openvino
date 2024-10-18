// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/shape_of.hpp"
#include "primitive_inst.h"

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<shape_of>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_CPU(shape_of, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_CPU(shape_of, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
