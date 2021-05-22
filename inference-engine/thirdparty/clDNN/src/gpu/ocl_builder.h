// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once

#include <cl2_wrapper.h>
#include <list>
#include <string>
#include <utility>
#include <vector>
#include <map>
#include "device_impl.h"

namespace cldnn {
namespace gpu {
struct configuration;

class ocl_builder {
private:
    const uint32_t device_type = CL_DEVICE_TYPE_GPU;  // cldnn supports only gpu devices
    const uint32_t device_vendor = 0x8086;  // Intel vendor
public:
    ocl_builder() = default;

    std::map<std::string, device_impl::ptr> get_available_devices(void* user_context, void* user_device) const;
    uint32_t get_device_type() const { return device_type; }
    uint32_t get_device_vendor() const { return device_vendor; }
private:
    bool does_device_match_config(bool out_of_order, const cl::Device& device) const;
    std::vector<device_impl::ptr> build_device_list(bool out_out_order) const;
    std::vector<device_impl::ptr> build_device_list_from_user_context(bool out_out_order, void* user_context) const;
    std::vector<device_impl::ptr> build_device_list_from_user_device(bool out_out_order, void* user_device) const;
};

}  // namespace gpu
}  // namespace cldnn
