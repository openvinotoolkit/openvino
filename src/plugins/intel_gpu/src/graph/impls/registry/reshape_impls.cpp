// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "program_node.h"
#include "primitive_inst.h"

namespace ov {
namespace intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<reshape>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_OCL(reshape, shape_types::static_shape),
        OV_GPU_GET_INSTANCE_OCL(reshape, shape_types::dynamic_shape,
            [](const program_node& node) {
                return node.can_be_optimized();
        }),
    };

    return impls;
}

}  // namespace intel_gpu
}  // namespace ov
