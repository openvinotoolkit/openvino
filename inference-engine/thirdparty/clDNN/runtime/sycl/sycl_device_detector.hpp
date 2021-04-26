// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/device.hpp"
#include "cldnn/runtime/engine_configuration.hpp"
#include "sycl_common.hpp"

#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>

namespace cldnn {
namespace sycl {

class sycl_device_detector {
private:
    const cl::sycl::info::device_type dev_type = cl::sycl::info::device_type::gpu;  // cldnn supports only gpu devices
    const uint32_t dev_vendor = 0x8086;  // Intel vendor
public:
    sycl_device_detector() = default;

    std::map<std::string, device::ptr> get_available_devices(runtime_types runtime_type, void* user_context, void* user_device) const;
private:
    bool does_device_match_config(const cl::sycl::device& device) const;
    std::vector<device::ptr> create_device_list(runtime_types runtime_type) const;
    std::vector<device::ptr> create_device_list_from_user_context(void* user_context) const;
    std::vector<device::ptr> create_device_list_from_user_device(void* user_device) const;
};

}  // namespace sycl
}  // namespace cldnn
