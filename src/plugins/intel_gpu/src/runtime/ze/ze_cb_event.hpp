// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event.hpp"

namespace cldnn {
namespace ze {


// L0 counter based event.
// Signaled state is inferred from the number of tasks completed on device.
// Resetting counter based event is not allowed.
// Signaling counter based event from host is not allowed.
// Can only be used with in-order command lists.
struct ze_cb_event : public ze_base_event {
public:
    // Take ownership of counter based event handle
    ze_cb_event(uint64_t queue_stamp, const ze_base_event_factory& factory, ze_event_handle_t ev)
    : ze_base_event(queue_stamp)
    , m_factory(factory)
    , m_event(ev) {
        // Ensure event handle is not null
        if (ev == nullptr) {
            OPENVINO_THROW("[GPU] Trying to create event with null handle");
        }
    }
    ze_cb_event(const ze_cb_event&) = delete;
    ze_cb_event& operator=(const ze_cb_event&) = delete;
    ~ze_cb_event();

    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;
    ze_event_handle_t get_handle() const override;
    std::optional<ze_kernel_timestamp_result_t> query_timestamp() override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

protected:
    const ze_base_event_factory& m_factory;
    ze_event_handle_t m_event;
};

}  // namespace ze
}  // namespace cldnn
