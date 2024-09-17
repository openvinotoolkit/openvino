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

// The priority return by this function impacts the order of devices reported by GPU plugin and devices enumeration
// Lower priority value means lower device ID
// Current behavior is: Intel iGPU < Intel dGPU < any other GPU
// Order of Intel dGPUs is undefined and depends on the OCL impl
// Order of other vendor GPUs is undefined and depends on the OCL impl
inline size_t get_device_priority(const cldnn::device_info& info) {
    if (info.vendor_id == cldnn::INTEL_VENDOR_ID && info.dev_type == cldnn::device_type::integrated_gpu) {
        return 0;
    } else if (info.vendor_id == cldnn::INTEL_VENDOR_ID) {
        return 1;
    } else {
        return std::numeric_limits<size_t>::max();
    }
}

inline std::vector<device::ptr> sort_devices(const std::vector<device::ptr>& devices_list) {
    std::vector<device::ptr> sorted_list = devices_list;
    std::stable_sort(sorted_list.begin(), sorted_list.end(), [](device::ptr d1,  device::ptr d2) {
        return get_device_priority(d1->get_info()) < get_device_priority(d2->get_info());
    });

    return sorted_list;
}

}  // namespace cldnn
