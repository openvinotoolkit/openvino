// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "registry.hpp"
#include "intel_gpu/primitives/fake_convert.hpp"
#include "primitive_inst.h"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<fake_convert>::get_implementations() {
    static const std::vector<std::shared_ptr<ImplementationManager>> impls = {
        OV_GPU_GET_INSTANCE_CPU(fake_convert, shape_types::static_shape)
        OV_GPU_GET_INSTANCE_CPU(fake_convert, shape_types::dynamic_shape)
    };

    return impls;
}

}  // namespace ov::intel_gpu
