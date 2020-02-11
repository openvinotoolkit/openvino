/*
// Copyright (c) 2018 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
// we want exceptions
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
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
    std::pair<std::string, device_impl::ptr> get_device(const uint32_t index,
        const cl::Device& dev_to_add, const cl_platform_id platform) const;
    std::pair<std::string, device_impl::ptr> get_device_shared(const uint32_t index,
        const cl::Device& dev_to_add, const cl_platform_id platform, void* user_device) const;
    bool does_device_match_config(bool out_of_order, const cl::Device& device) const;
    std::map<std::string, device_impl::ptr> build_device_list(bool out_out_order) const;
    std::map<std::string, device_impl::ptr> build_device_list_from_user_context(bool out_out_order, void* user_context) const;
    std::map<std::string, device_impl::ptr> build_device_list_from_user_device(bool out_out_order, void* user_device) const;
};

}  // namespace gpu
}  // namespace cldnn
