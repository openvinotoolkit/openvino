// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device_info.hpp"
#include "memory_caps.hpp"

#include <memory>

namespace cl {
class Device;
}

namespace cldnn {

/// @brief Represents detected GPU device object. Use device_query to get list of available objects.
struct device {
public:
    using ptr = std::shared_ptr<device>;
    virtual device_info get_info() const = 0;
    virtual memory_capabilities get_mem_caps() const = 0;

    virtual bool is_same(const device::ptr other) = 0;
    virtual const cl::Device& get_device() const = 0;

    virtual ~device() = default;
};

}  // namespace cldnn
