// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/engine_configuration.hpp"

namespace cldnn {

engine_types get_default_engine_type() {
#ifdef OV_GPU_WITH_ZE_RT
    return engine_types::ze;
#elif defined(OV_GPU_WITH_OCL_RT)
    return engine_types::ocl;
#elif defined(OV_GPU_WITH_SYCL_RT)
    return engine_types::sycl;
#else
    #error "Expected OpenVINO GPU runtime macros to be defined"
#endif
}

runtime_types get_default_runtime_type() {
#ifdef OV_GPU_WITH_ZE_RT
    return runtime_types::ze;
#elif defined(OV_GPU_WITH_OCL_RT)
    return runtime_types::ocl;
#elif defined(OV_GPU_WITH_SYCL_RT)
    return runtime_types::sycl;
#else
    #error "Expected OpenVINO GPU runtime macros to be defined"
#endif
}

std::string_view to_cache_tag(runtime_types type) {
    // Stable strings - do not change (see header: cache-compatibility constant).
    switch (type) {
    case runtime_types::ocl: return "OCL";
    case runtime_types::ze: return "ZE";
    case runtime_types::sycl: return "SYCL";
    default: return "UNKNOWN";
    }
}

std::string_view get_runtime_cache_tag() {
    return to_cache_tag(get_default_runtime_type());
}
}  // namespace cldnn
