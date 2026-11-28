// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "sycl_common.hpp"

#include <memory>
#include <mutex>

namespace cldnn {
namespace ocl { struct ocl_device; }
namespace sycl {

struct sycl_device : public device {
public:
    sycl_device(const ::sycl::device dev, const ::sycl::context& ctx, const ::sycl::platform& platform);

    const device_info& get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    void initialize() override;
    bool is_initialized() const override { return _is_initialized; };

    const ::sycl::device& get_device() const { return _device; }
    const ::sycl::context& get_context() const { return _context; }
    const ::sycl::platform& get_platform() const { return _platform; }

    // The OpenCL device that is the same GPU as this SYCL device, initialized on first use, or nullptr. Only NVIDIA
    // SYCL devices have one: their kernel builder compiles OpenCL C kernels with clBuildProgram on it. It is looked up
    // once, when the device is created, and shared by every kernel builder of the device.
    std::shared_ptr<ocl::ocl_device> get_ocl_device() const;

    bool is_same(const device::ptr other) override;

    void set_mem_caps(const memory_capabilities& memory_capabilities) override;

    ~sycl_device() = default;

private:
    ::sycl::context _context;
    ::sycl::device _device;
    ::sycl::platform _platform;
    device_info _info;
    memory_capabilities _mem_caps;
    bool _is_initialized = false;
    std::shared_ptr<ocl::ocl_device> _ocl_device;
    mutable std::once_flag _ocl_device_init;
};

}  // namespace sycl
}  // namespace cldnn
