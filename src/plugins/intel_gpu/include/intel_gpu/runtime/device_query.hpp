// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device.hpp"
#include "engine_configuration.hpp"

#include <map>
#include <string>
#include <algorithm>
#include <vector>

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
                          int target_tile_id = -1,
                          bool initialize_devices = false);

    /// @brief Create device query with default values for engine type and runtime type
    explicit device_query(void* user_context = nullptr,
                          void* user_device = nullptr,
                          int ctx_device_id = 0,
                          int target_tile_id = -1,
                          bool initialize_devices = false);

    std::map<std::string, device::ptr> get_available_devices() const {
        return _available_devices;
    }

    ~device_query() = default;
private:
    std::map<std::string, device::ptr> _available_devices;
};

// One device seen by the lightweight enumeration below.
struct lightweight_device {
    std::string map_id;    ///< Detector's own id for the device (this library's ".N").
    device_info info;      ///< Device info populated without engine/context construction.
};

// Cheap device enumeration for pre-construction dispatch: lists the devices this build's
// runtime can serve with device_info populated, WITHOUT building an engine/context.
std::vector<lightweight_device> lightweight_enumerate() noexcept;

}  // namespace cldnn
