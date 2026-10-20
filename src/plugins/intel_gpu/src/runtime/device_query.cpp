// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/device_query.hpp"
#include "ocl/ocl_device_detector.hpp"
#include "ze/ze_device_detector.hpp"

#ifdef OV_GPU_WITH_SYCL_RT
#include "sycl/sycl_device_detector.hpp"
#endif

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
#if defined(OV_GPU_WITH_OCL_RT) || defined(OV_GPU_WITH_ZE_RT)
    case runtime_types::ocl: {
        OPENVINO_ASSERT(engine_type == engine_types::ocl || engine_type == engine_types::sycl);
        ocl::ocl_device_detector ocl_detector;
        _available_devices = ocl_detector.get_available_devices(user_context, user_device, ctx_device_id, target_tile_id, initialize_devices);
#ifdef OV_GPU_WITH_ZE_RT
        // If running with ZE runtime, convert found OCL devices to ZE devices
        std::map<std::string, device::ptr> ze_devices;
        for (auto& device : _available_devices) {
            auto ze_device = ze::create_ze_device_from_ocl_device(device.second, initialize_devices);
            ze_devices[device.first] = ze_device;
        }
        _available_devices = std::move(ze_devices);
#endif
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
#ifdef OV_GPU_WITH_SYCL_RT
    case runtime_types::sycl: {
        OPENVINO_ASSERT(engine_type == engine_types::sycl);
        sycl::sycl_device_detector sycl_detector;
        _available_devices = sycl_detector.get_available_devices(user_context, user_device, ctx_device_id, target_tile_id);
        break;
    }
#endif
    default: OPENVINO_THROW("[GPU] Unsupported engine/runtime types in device_query");
    }
}

std::vector<lightweight_device> lightweight_enumerate() noexcept {
    std::vector<lightweight_device> result;
    try {
        // Reuse the standard detector for this build's runtime, but do NOT initialize
        // devices (no engine/context) - only the info populated at detection is read.
        device_query query(nullptr, nullptr, 0, -1, /*initialize_devices=*/false);
        for (const auto& [map_id, dev] : query.get_available_devices()) {
            result.push_back({map_id, dev->get_info()});
        }
    } catch (...) {
        result.clear();  // Any failure -> "I serve nothing"; never throw from the probe.
    }
    return result;
}
}  // namespace cldnn
