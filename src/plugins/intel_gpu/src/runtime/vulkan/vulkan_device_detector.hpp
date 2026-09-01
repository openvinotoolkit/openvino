// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <map>
#include <string>

#include "intel_gpu/runtime/device.hpp"

namespace cldnn {
namespace vulkan {

class vulkan_device_detector {
public:
    std::map<std::string, device::ptr> get_available_devices(void* user_context,
                                                             void* user_device,
                                                             int ctx_device_id,
                                                             int target_tile_id,
                                                             bool initialize_devices) const;
};

}  // namespace vulkan
}  // namespace cldnn
