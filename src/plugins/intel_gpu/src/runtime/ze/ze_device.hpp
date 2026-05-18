// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/device.hpp"
#include "ze_common.hpp"
#include "ze_resource.hpp"

namespace cldnn {
namespace ze {

struct ze_device : public device {
public:
    ze_device(ze_driver_resource driver, ze_device_resource device, bool initialize = true);
    ze_device(const ze_device &other) = delete;
    ze_device& operator=(const ze_device &other) = delete;

    const device_info& get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    void initialize() override;
    bool is_initialized() const override;

    const ze_driver_resource& get_driver() const { return _driver; }
    const ze_device_resource& get_device() const { return _device; }
    const ze_context_resource& get_context() const { return _context; }

    bool is_same(const device::ptr other) override;
    void set_mem_caps(const memory_capabilities& memory_capabilities) override;

    ~ze_device() = default;

private:
    ze_driver_resource _driver;
    ze_device_resource _device;
    ze_context_resource _context;

    device_info _info;
    memory_capabilities _mem_caps;
};

}  // namespace ze
}  // namespace cldnn
