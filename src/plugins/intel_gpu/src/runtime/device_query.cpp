// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/device_query.hpp"
#include "ocl/ocl_device_detector.hpp"
#include "ze/ze_device_detector.hpp"

#include <map>

namespace cldnn {
int device_query::device_id = -1;
device_query::device_query(engine_types engine_type,
                           runtime_types runtime_type,
                           void* user_context,
                           void* user_device,
                           int ctx_device_id,
                           int target_tile_id) {
    switch (runtime_type) {
    case runtime_types::ocl: {
        OPENVINO_ASSERT(engine_type == engine_types::ocl || engine_type == engine_types::sycl);
        ocl::ocl_device_detector ocl_detector;
        _available_devices = ocl_detector.get_available_devices(user_context, user_device, ctx_device_id, target_tile_id);
        break;
    }
#ifdef OV_GPU_WITH_ZE_RT
    case runtime_types::ze: {
        OPENVINO_ASSERT(engine_type == engine_types::ze);
        ze::ze_device_detector ze_detector;
        _available_devices = ze_detector.get_available_devices(user_context, user_device, ctx_device_id, target_tile_id);
        break;
    }
#endif
    default: OPENVINO_THROW("[GPU] Unsupported engine/runtime types in device_query");
    }
}
}  // namespace cldnn
