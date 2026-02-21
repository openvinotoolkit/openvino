// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "ze_base_event_factory.hpp"

#include <ze_api.h>
#include <chrono>
#include <optional>

namespace cldnn {
namespace ze {

// Base interface for Level Zero events
struct ze_base_event : public event {
public:
    explicit ze_base_event(uint64_t queue_stamp)
    : event()
    , m_queue_stamp(queue_stamp) { }
    uint64_t get_queue_stamp() const { return m_queue_stamp; }
    void set_queue_stamp(uint64_t val) { m_queue_stamp = val; }

    virtual ze_event_handle_t get_handle() const = 0;
    virtual std::optional<ze_kernel_timestamp_result_t> query_timestamp() = 0;

protected:
    uint64_t m_queue_stamp = 0;

    static std::chrono::nanoseconds timestamp_to_duration(const device_info &info, const ze_kernel_timestamp_data_t& timestamp) {
        constexpr double NS_IN_SEC = 1000000000.0;
        const double timestamp_freq = NS_IN_SEC / info.timer_resolution;
        const uint64_t timestamp_max_value = ~(-1L << info.kernel_timestamp_valid_bits);

        auto d = (timestamp.kernelEnd >= timestamp.kernelStart) ?
            (timestamp.kernelEnd - timestamp.kernelStart) * timestamp_freq
            : ((timestamp_max_value - timestamp.kernelStart) + timestamp.kernelEnd + 1) * timestamp_freq;
        return std::chrono::nanoseconds(static_cast<uint64_t>(d));
    }
};

}  // namespace ze
}  // namespace cldnn
