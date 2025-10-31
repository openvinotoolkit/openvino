// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event.hpp"
#include "ze_event_pool.hpp"

namespace cldnn {
namespace ze {

// L0 event. Can be either in signaled state or not signaled state.
struct ze_event : public ze_base_event {
public:
    // Take ownership of event handle
    ze_event(uint64_t queue_stamp, const ze_base_event_factory& factory, ze_event_handle_t ev, std::shared_ptr<ze_event_pool> event_pool)
        : ze_base_event(queue_stamp)
        , m_event_pool(event_pool)
        , m_factory(factory)
        , m_event(ev) {
            // Ensure event handle is not null
            if (ev == nullptr) {
                OPENVINO_THROW("[GPU] Trying to create event with null handle");
            }
        }
    ze_event(const ze_event &) = delete;
    ze_event& operator=(const ze_event &) = delete;
    ~ze_event();
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

    std::shared_ptr<ze_event_pool> m_event_pool;
    const ze_base_event_factory& m_factory;
    ze_event_handle_t m_event;
};

}  // namespace ze
}  // namespace cldnn
