// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/device_query.hpp"

#include "intel_gpu/runtime/runtime_backend_registry.hpp"

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
    _available_devices =
        runtime_backend_registry::query_devices(engine_type, runtime_type, user_context, user_device, ctx_device_id, target_tile_id, initialize_devices);
}
}  // namespace cldnn
