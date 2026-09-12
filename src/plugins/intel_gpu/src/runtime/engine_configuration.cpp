// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/engine_configuration.hpp"

#include "intel_gpu/runtime/runtime_backend_registry.hpp"

namespace cldnn {

engine_types get_default_engine_type() {
    return runtime_backend_registry::default_backend().engine_type;
}

runtime_types get_default_runtime_type() {
    return runtime_backend_registry::default_backend().runtime_type;
}

std::string_view to_cache_tag(runtime_types type) {
    // Stable strings - do not change (see header: cache-compatibility constant).
    switch (type) {
    case runtime_types::ocl: return "OCL";
    case runtime_types::ze: return "ZE";
    case runtime_types::sycl: return "SYCL";
    case runtime_types::vulkan: return "VULKAN";
    default: return "UNKNOWN";
    }
}

std::string_view get_runtime_cache_tag() {
    return to_cache_tag(get_default_runtime_type());
}
}  // namespace cldnn
