// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/engine_configuration.hpp"

namespace cldnn {

engine_types get_default_engine_type() {
#ifdef OV_GPU_DEFAULT_ZE_RT
    return engine_types::ze;
#elif defined(OV_GPU_DEFAULT_OCL_RT)
    return engine_types::ocl;
#elif defined(OV_GPU_DEFAULT_SYCL_RT)
    return engine_types::sycl;
#elif defined(OV_GPU_DEFAULT_VULKAN_RT)
    return engine_types::vulkan;
#else
    #error "Expected an OpenVINO default GPU runtime macro to be defined"
#endif
}

runtime_types get_default_runtime_type() {
#ifdef OV_GPU_DEFAULT_ZE_RT
    return runtime_types::ze;
#elif defined(OV_GPU_DEFAULT_OCL_RT)
    return runtime_types::ocl;
#elif defined(OV_GPU_DEFAULT_SYCL_RT)
    return runtime_types::sycl;
#elif defined(OV_GPU_DEFAULT_VULKAN_RT)
    return runtime_types::vulkan;
#else
    #error "Expected an OpenVINO default GPU runtime macro to be defined"
#endif
}
}  // namespace cldnn
