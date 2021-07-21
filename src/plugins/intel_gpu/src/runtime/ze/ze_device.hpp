// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/engine.hpp"
#include "cldnn/runtime/device.hpp"
#include "ze_common.hpp"

#include <map>
#include <string>
#include <vector>
#include <algorithm>

namespace cldnn {
namespace ze {

struct ze_device : public device {
public:
    ze_device(ze_driver_handle_t driver, ze_device_handle_t device);

    device_info get_info() const override { return _info; }
    memory_capabilities get_mem_caps() const override { return _mem_caps; }

    const ze_driver_handle_t get_driver() const { return _driver; }
    const ze_device_handle_t get_device() const { return _device; }
    const ze_context_handle_t get_context() const { return _context; }

    ~ze_device();

private:
    ze_driver_handle_t _driver = nullptr;
    ze_device_handle_t _device = nullptr;
    ze_context_handle_t _context = nullptr;

    device_info _info;
    memory_capabilities _mem_caps;
};

}  // namespace ze
}  // namespace cldnn
