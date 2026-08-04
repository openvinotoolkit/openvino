// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/device_query.hpp"
#include "vulkan/vk_device_detector.hpp"

#include <map>

namespace cldnn {
int device_query::device_id = -1;

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
#ifdef OV_GPU_WITH_VULKAN_RT
    case runtime_types::vulkan: {
        OPENVINO_ASSERT(engine_type == engine_types::vulkan);
        vk::vk_device_detector vk_detector;
        _available_devices = vk_detector.get_available_devices(user_context, user_device, ctx_device_id, target_tile_id, initialize_devices);
        break;
    }
#endif
    default: OPENVINO_THROW("[GPU] Unsupported engine/runtime types in device_query");
    }
}
}  // namespace cldnn
