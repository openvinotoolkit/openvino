// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "device_info.hpp"
#include "memory_caps.hpp"
#include "layout.hpp"

#include <memory>

namespace cldnn {

const uint32_t INTEL_VENDOR_ID = 0x8086;

/// @brief Represents detected GPU device object. Use device_query to get list of available objects.
struct device {
public:
    using ptr = std::shared_ptr<device>;
    virtual const device_info& get_info() const = 0;
    virtual memory_capabilities get_mem_caps() const = 0;

    virtual bool is_same(const device::ptr other) = 0;

    float get_gops(cldnn::data_types dt) const;
    bool use_unified_shared_memory() const;

    virtual ~device() = default;
};

}  // namespace cldnn
