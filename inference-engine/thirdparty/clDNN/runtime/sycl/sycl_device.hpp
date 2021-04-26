// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/device.hpp"
#include "sycl_common.hpp"

#include <map>
#include <string>
#include <vector>
#include <algorithm>

namespace cldnn {
namespace sycl {

struct sycl_device : public device {
public:
    sycl_device(const cl::sycl::device& dev, const cl::sycl::context& ctx);

    device_info get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    const cl::sycl::device& get_device() const { return _device; }
    const cl::sycl::context& get_context() const { return _context; }

    cl_device_id get_ocl_device() const { return _device.get(); }

    ~sycl_device() = default;

private:
    cl::sycl::context _context;
    cl::sycl::device _device;
    device_info _info;
    memory_capabilities _mem_caps;
};

}  // namespace sycl
}  // namespace cldnn
