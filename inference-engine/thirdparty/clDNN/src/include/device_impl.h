/*
// Copyright (c) 2019 Intel Corporation
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
#define CL_HPP_ENABLE_EXCEPTIONS
#define CL_HPP_MINIMUM_OPENCL_VERSION 120
#define CL_HPP_TARGET_OPENCL_VERSION 120
#include <cl2_wrapper.h>
#include "gpu/device_info.h"
#include "api/device.hpp"
#include "refcounted_obj.h"
#include "gpu/configuration.h"

#include <map>
#include <string>

namespace cldnn {

struct device_impl : public refcounted_obj<device_impl> {
public:
    explicit device_impl(const cl::Device dev, const cl::Context& ctx, const cl_platform_id platform, const gpu::device_info_internal& info)
        : _device(dev), _context(ctx), _platform(platform), _info(info)
    {}

    gpu::device_info_internal get_info() const { return _info; }
    cl::Device get_device() const { return _device; }
    cl::Context get_context() const { return _context; }
    cl_platform_id get_platform() const { return _platform; }

    ~device_impl() = default;
private:
    cl::Device _device;
    cl::Context _context;
    cl_platform_id _platform;
    gpu::device_info_internal _info;
};

struct device_query_impl : public refcounted_obj<device_query_impl> {
public:
    explicit device_query_impl(void* user_context = nullptr, void* user_device = nullptr);

    std::map<std::string, device_impl::ptr> get_available_devices() const {
        return _available_devices;
    }

    ~device_query_impl() = default;
private:
    std::map<std::string, device_impl::ptr> _available_devices;
};

}  // namespace cldnn
