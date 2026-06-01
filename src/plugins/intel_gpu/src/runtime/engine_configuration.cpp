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
#elif defined(OV_GPU_WITH_SYCL)
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
#else
    #error "Expected OpenVINO GPU runtime macros to be defined"
#endif
}
}  // namespace cldnn
