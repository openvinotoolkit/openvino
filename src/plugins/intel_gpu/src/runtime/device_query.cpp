// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/device_query.hpp"
#include "ocl/ocl_device_detector.hpp"
#include "ze/ze_device_detector.hpp"

#include <map>

namespace cldnn {
int device_query::device_id = -1;

engine_types device_query::get_default_engine_type() {
    auto engine_type = engine_types::ocl;
#ifdef OV_GPU_WITH_ZE_RT
    engine_type = engine_types::ze;
#endif
#ifdef OV_GPU_WITH_OCL_RT
    engine_type = engine_types::ocl;
#endif
#ifdef OV_GPU_WITH_SYCL
    engine_type = engine_types::sycl;
#endif
    return engine_type;
}
runtime_types device_query::get_default_runtime_type() {
    auto rt_type = runtime_types::ocl;
#ifdef OV_GPU_WITH_ZE_RT
    rt_type = runtime_types::ze;
#endif
#ifdef OV_GPU_WITH_OCL_RT
    rt_type = runtime_types::ocl;
#endif
    return rt_type;
}

device_query::device_query(void* user_context,
                           void* user_device,
                           int ctx_device_id,
                           int target_tile_id,
                           bool initialize_devices)
    : device_query(get_default_engine_type(),
        get_default_runtime_type(),
        user_context,
        user_device,
        ctx_device_id,
        target_tile_id,
        initialize_devices) {}

device_query::device_query(engine_types engine_type,
                           runtime_types runtime_type,
                           void* user_context,
                           void* user_device,
                           int ctx_device_id,
                           int target_tile_id,
                           bool initialize_devices) {
    switch (runtime_type) {
#ifdef OV_GPU_WITH_OCL_RT
    case runtime_types::ocl: {
        OPENVINO_ASSERT(engine_type == engine_types::ocl || engine_type == engine_types::sycl);
        ocl::ocl_device_detector ocl_detector;
        _available_devices = ocl_detector.get_available_devices(user_context, user_device, ctx_device_id, target_tile_id, initialize_devices);
        break;
    }
#endif
#ifdef OV_GPU_WITH_ZE_RT
    case runtime_types::ze: {
        OPENVINO_ASSERT(engine_type == engine_types::ze);
        ze::ze_device_detector ze_detector;
        _available_devices = ze_detector.get_available_devices(user_context, user_device, ctx_device_id, target_tile_id, initialize_devices);
        break;
    }
#endif
    default: OPENVINO_THROW("[GPU] Unsupported engine/runtime types in device_query");
    }
}
}  // namespace cldnn
