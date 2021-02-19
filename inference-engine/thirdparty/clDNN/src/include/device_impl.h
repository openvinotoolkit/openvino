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
#include <cl2_wrapper.h>
#include "gpu/device_info.h"
#include "api/device.hpp"
#include "refcounted_obj.h"
#include "gpu/configuration.h"

#include <map>
#include <string>
#include <vector>
#include <algorithm>

namespace cldnn {
enum class allocation_type {
    unknown,     // Not specified (i.e simple_attached_memory class).
    cl_mem,      // Use standard OpenCL cl_mem allocations.
    usm_host,    // Accessible only by host. Not Migratable
    usm_shared,  // Accessible by host and device. Migrtable.
    usm_device,  // Accessible only by device. Not migratable.
};

struct device_impl;

class memory_capabilities {
public:
    memory_capabilities(bool support_usm, const cl::Device& cl_dev) : _caps({ allocation_type::cl_mem }) {
        if (support_usm) {
            if (does_device_support(CL_DEVICE_HOST_MEM_CAPABILITIES_INTEL, cl_dev)) {
                _caps.push_back(allocation_type::usm_host);
            }
            if (does_device_support(CL_DEVICE_SINGLE_DEVICE_SHARED_MEM_CAPABILITIES_INTEL, cl_dev)) {
                _caps.push_back(allocation_type::usm_shared);
            }
            if (does_device_support(CL_DEVICE_DEVICE_MEM_CAPABILITIES_INTEL, cl_dev)) {
                 _caps.push_back(allocation_type::usm_device);
            }
        }
    }

    bool supports_usm() const { return find_in_caps(allocation_type::cl_mem) && _caps.size() > 1; }
    bool support_allocation_type(allocation_type type) const { return find_in_caps(type); }

    static bool is_usm_type(allocation_type type) {
        if (type == allocation_type::usm_host ||
            type == allocation_type::usm_shared ||
            type == allocation_type::usm_device)
            return true;
        return false;
    }

private:
    std::vector<allocation_type> _caps;

    bool does_device_support(int32_t param, const cl::Device& device) {
        cl_device_unified_shared_memory_capabilities_intel capabilities;
        auto err = clGetDeviceInfo(device.get(), param, sizeof(cl_device_unified_shared_memory_capabilities_intel), &capabilities, NULL);
        if (err) throw std::runtime_error("[CLDNN ERROR]. clGetDeviceInfo error " + std::to_string(err));
        return !((capabilities & CL_UNIFIED_SHARED_MEMORY_ACCESS_INTEL) == 0u);
    }

    bool find_in_caps(const allocation_type& type) const {
        return std::find_if(_caps.begin(), _caps.end(), [&](const allocation_type& t) { return t == type; }) != _caps.end();
    }
};


struct device_impl : public refcounted_obj<device_impl> {
public:
    explicit device_impl(const cl::Device dev, const cl::Context& ctx, const cl_platform_id platform, const gpu::device_info_internal& info)
        : _context(ctx), _device(dev), _platform(platform), _info(info), _mem_caps(_info.supports_usm, _device) { }

    gpu::device_info_internal get_info() const { return _info; }
    cl::Device get_device() const { return _device; }
    cl::Context get_context() const { return _context; }
    cl_platform_id get_platform() const { return _platform; }
    memory_capabilities mem_caps() const { return _mem_caps; }

    ~device_impl() = default;

private:
    cl::Context _context;
    cl::Device _device;
    cl_platform_id _platform;
    gpu::device_info_internal _info;
    memory_capabilities _mem_caps;
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
