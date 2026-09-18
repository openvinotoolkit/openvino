// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event.hpp"
#include "ze_resource.hpp"

namespace cldnn {
namespace ze {


// ze counter based event.
// Signaled state is inferred from the number of tasks completed on device.
// Resetting counter based event is not allowed.
// Start in signaled state and signaling from host is not allowed.
// Can only be used with in-order command lists.
struct ze_counter_based_event : public ze_base_event {
public:
    ze_counter_based_event(uint64_t queue_stamp, const ze_base_event_factory& factory, ze_event_resource ev)
    : ze_base_event(queue_stamp)
    , m_factory(factory)
    , m_event(ev) {
        OPENVINO_ASSERT(!m_event.is_empty(), "[GPU] Attempt to create counter based event with empty holder");
    }

    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;
    ze_event_handle_t get_handle() const override;
    std::optional<ze_kernel_timestamp_result_t> query_timestamp() override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

protected:
    const ze_base_event_factory& m_factory;
    ze_event_resource m_event;
};

}  // namespace ze
}  // namespace cldnn
