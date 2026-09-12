// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/primitives/implementation_desc.hpp"
#include "intel_gpu/runtime/engine_configuration.hpp"

namespace cldnn {

class runtime_implementation_policy final {
public:
    static bool allows(runtime_types runtime_type, impl_types implementation_type, bool is_shape_flow) {
        if (implementation_type == impl_types::common) {
            return true;
        }
        if (implementation_type == impl_types::cpu) {
            return runtime_type != runtime_types::vulkan || is_shape_flow;
        }

        switch (runtime_type) {
        case runtime_types::ocl:
        case runtime_types::ze:
            return implementation_type == impl_types::ocl || implementation_type == impl_types::onednn || implementation_type == impl_types::cm;
        case runtime_types::sycl:
            return implementation_type == impl_types::sycl || implementation_type == impl_types::onednn;
        case runtime_types::vulkan:
            return implementation_type == impl_types::vulkan;
        }
        return false;
    }
};

}  // namespace cldnn
