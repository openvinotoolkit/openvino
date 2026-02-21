// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event.hpp"

namespace cldnn {
namespace ze {


// Event that does not have underlying Level Zero event object.
// It is always in signaled state.
struct ze_empty_event : public ze_base_event {
public:
    ze_empty_event(uint64_t queue_stamp)
    : ze_base_event(queue_stamp) { }

    void wait_impl() override { }
    void set_impl() override { }
    bool is_set_impl() override { return true; }
    ze_event_handle_t get_handle() const override { return nullptr; }
    std::optional<ze_kernel_timestamp_result_t> query_timestamp() override { return std::nullopt; }
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override {
        return true;
    }
};

}  // namespace ze
}  // namespace cldnn
