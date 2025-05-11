// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device.hpp"
#include "engine_configuration.hpp"

#include <map>
#include <string>
#include <algorithm>

namespace cldnn {

// Fetches all available gpu devices with specific runtime and engine types and (optionally) user context/device handles
struct device_query {
public:
    static int device_id;
    explicit device_query(engine_types engine_type,
                          runtime_types runtime_type,
                          void* user_context = nullptr,
                          void* user_device = nullptr,
                          int ctx_device_id = 0,
                          int target_tile_id = -1);

    std::map<std::string, device::ptr> get_available_devices() const {
        return _available_devices;
    }

    ~device_query() = default;
private:
    std::map<std::string, device::ptr> _available_devices;
};
}  // namespace cldnn
