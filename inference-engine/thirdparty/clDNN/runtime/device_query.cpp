// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "cldnn/runtime/device_query.hpp"
#include "ocl/ocl_device_detector.hpp"

#ifdef CLDNN_WITH_SYCL
#include "sycl/sycl_device_detector.hpp"
#endif

#include <map>
#include <string>
namespace cldnn {

device_query::device_query(engine_types engine_type,
                           runtime_types runtime_type,
                           void* user_context,
                           void* user_device,
                           int ctx_device_id,
                           int target_tile_id) {
    switch (engine_type) {
#ifdef CLDNN_WITH_SYCL
    case engine_types::sycl: {
        sycl::sycl_device_detector sycl_detector;
        auto sycl_devices = sycl_detector.get_available_devices(runtime_type, user_context, user_device);
        _available_devices.insert(sycl_devices.begin(), sycl_devices.end());
        break;
    }
#endif
    case engine_types::ocl: {
        if (runtime_type != runtime_types::ocl)
            throw std::runtime_error("Unsupported runtime type for ocl engine");

        ocl::ocl_device_detector ocl_detector;
        _available_devices = ocl_detector.get_available_devices(user_context, user_device, ctx_device_id, target_tile_id);
        break;
    }
    default: throw std::runtime_error("Unsupported engine type in device_query");
    }

    if (_available_devices.empty()) {
        throw std::runtime_error("No suitable devices found for requested engine and runtime types");
    }
}
}  // namespace cldnn
