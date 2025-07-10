// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/arg_max_min.hpp"
#include "arg_max_min_inst.h"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<arg_max_min>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_OCL(arg_max_min, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(arg_max_min, shape_types::dynamic_shape,
            [](const program_node& node) {
                return node.as<arg_max_min>().get_primitive()->top_k != 0;
        })
    };

    return impls;
}

}  // namespace ov::intel_gpu
