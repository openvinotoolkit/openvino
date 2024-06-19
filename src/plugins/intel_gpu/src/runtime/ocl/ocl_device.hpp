// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/engine.hpp"
#include "intel_gpu/runtime/device.hpp"
#include "ocl_common.hpp"

#include <map>
#include <string>
#include <vector>
#include <algorithm>

namespace cldnn {
namespace ocl {

struct ocl_device : public device {
public:
    ocl_device(const cl::Device dev, const cl::Context& ctx, const cl::Platform& platform);

    device_info get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    const cl::Device& get_device() const { return _device; }
    cl::Device& get_device() { return _device; }
    const cl::Context& get_context() const { return _context; }
    const cl::Platform& get_platform() const { return _platform; }

    bool is_same(const device::ptr other) override;

    ~ocl_device() = default;

private:
    cl::Context _context;
    cl::Device _device;
    cl::Platform _platform;
    device_info _info;
    memory_capabilities _mem_caps;
};

}  // namespace ocl
}  // namespace cldnn
