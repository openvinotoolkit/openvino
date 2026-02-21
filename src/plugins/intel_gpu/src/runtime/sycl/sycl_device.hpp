// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "sycl_common.hpp"

namespace cldnn {
namespace sycl {

struct sycl_device : public device {
public:
    sycl_device(const ::sycl::device dev, const ::sycl::context& ctx, const ::sycl::platform& platform);

    const device_info& get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    void initialize() override;
    bool is_initialized() const override { return _is_initialized; };

    const ::sycl::device& get_device() const { return _device; }
    ::sycl::device& get_device() { return _device; }
    const ::sycl::context& get_context() const { return _context; }
    const ::sycl::platform& get_platform() const { return _platform; }
    // const ::sycl::usm_helper& get_usm_helper() const { return *_usm_helper; }

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
    // std::unique_ptr<::sycl::usm_helper> _usm_helper;
};

}  // namespace sycl
}  // namespace cldnn
