// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"

#include <string>
#include <vector>
#include <map>

namespace cldnn {
namespace ze {

class ze_device_detector {
public:
    ze_device_detector() = default;

    std::map<std::string, device::ptr> get_available_devices(void* user_context,
                                                             void* user_device,
                                                             int ctx_device_id,
                                                             int target_tile_id) const;
private:
    std::vector<device::ptr> create_device_list() const;
    std::vector<device::ptr> create_device_list_from_user_context(void* user_context, int ctx_device_id = 0) const;
    std::vector<device::ptr> create_device_list_from_user_device(void* user_device) const;
};

}  // namespace ze
}  // namespace cldnn
