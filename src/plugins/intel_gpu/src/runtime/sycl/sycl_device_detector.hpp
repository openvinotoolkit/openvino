// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "intel_gpu/runtime/device.hpp"

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

namespace cldnn {
namespace sycl {

class sycl_device_detector {
public:
    sycl_device_detector() = default;

    static std::map<std::string, device::ptr> get_available_devices(void* user_context, void* user_device, int ctx_device_id = 0, int target_tile_id = -1);

private:
    static std::vector<device::ptr> create_device_list();
    static std::vector<device::ptr> create_device_list_from_user_context(void* user_context, int ctx_device_id = 0);
    static std::vector<device::ptr> create_device_list_from_user_device(void* user_device);
};

}  // namespace sycl
}  // namespace cldnn
