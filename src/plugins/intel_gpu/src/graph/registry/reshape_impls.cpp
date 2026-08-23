// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "backend_implementation_registry.hpp"
#include "intel_gpu/primitives/reshape.hpp"
#include "primitive_inst.h"
#include "registry.hpp"

namespace ov::intel_gpu {

using namespace cldnn;

const std::vector<std::shared_ptr<cldnn::ImplementationManager>>& Registry<reshape>::get_implementations() {
    static const auto impls = compose_backend_implementations<reshape>({OV_GPU_GET_INSTANCE_OCL(reshape, shape_types::static_shape)});

    return impls;
}

}  // namespace ov::intel_gpu
