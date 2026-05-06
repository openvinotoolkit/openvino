// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event.hpp"
#include "ze_holder.hpp"

namespace cldnn {
namespace ze {

// L0 event. Can be either in signaled state or not signaled state.
struct ze_event : public ze_base_event {
public:
    ze_event(uint64_t queue_stamp, const ze_base_event_factory& factory, ze_holder<ze_resource_type::event> ev)
        : ze_base_event(queue_stamp)
        , m_factory(factory)
        , m_event(ev) {
            OPENVINO_ASSERT(!m_event.is_empty(), "[GPU] Attempt to create event with empty holder");
        }
    void reset() override;

    std::optional<ze_kernel_timestamp_result_t> query_timestamp() override;
    ze_event_handle_t get_handle() const override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

protected:
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;
    // TODO: Implement add_event_handler_impl
    // bool add_event_handler_impl(event_handler, void*) override;

    const ze_base_event_factory& m_factory;
    ze_holder<ze_resource_type::event> m_event;
};

}  // namespace ze
}  // namespace cldnn
