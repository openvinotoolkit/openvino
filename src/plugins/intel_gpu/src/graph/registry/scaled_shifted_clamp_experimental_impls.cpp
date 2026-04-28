// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "predicates.hpp"
#include "registry.hpp"
#include "intel_gpu/primitives/scaled_shifted_clamp_experimental.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>&
Registry<scaled_shifted_clamp_experimental>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_OCL(scaled_shifted_clamp_experimental, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_OCL(scaled_shifted_clamp_experimental, shape_types::dynamic_shape)
    };
    return impls;
}

}  // namespace ov::intel_gpu
