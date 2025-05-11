// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

namespace cldnn {
namespace ocl {

class ocl_device_detector {
public:
    ocl_device_detector() = default;

    std::map<std::string, device::ptr> get_available_devices(void *user_context, void *user_device, int ctx_device_id = 0, int target_tile_id = -1) const;

    static std::vector<device::ptr> sort_devices(const std::vector<device::ptr>& devices_list);

private:
    std::vector<device::ptr> create_device_list() const;
    std::vector<device::ptr> create_device_list_from_user_context(void* user_context, int ctx_device_id = 0) const;
    std::vector<device::ptr> create_device_list_from_user_device(void* user_device) const;
};

}  // namespace ocl
}  // namespace cldnn
