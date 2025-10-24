// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event.hpp"
#include "ze_engine.hpp"

#include <vector>

namespace cldnn {
namespace ze {

struct ze_events : public ze_base_event {
public:
    ze_events(std::vector<event::ptr> const& ev, const ze_engine &engine)
        : ze_base_event(0)
        , m_engine(engine) {
        process_events(ev);
    }

    void reset() override {
        event::reset();
        m_events.clear();
    }

    std::optional<ze_kernel_timestamp_result_t> query_timestamp() override { return std::nullopt; }
    ze_event_handle_t get_handle() const { return m_last_event; }
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

protected:
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;

    void process_events(const std::vector<event::ptr>& ev) {
        for (size_t i = 0; i < ev.size(); i++) {
            auto multiple_events = dynamic_cast<ze_events*>(ev[i].get());
            if (multiple_events) {
                for (size_t j = 0; j < multiple_events->m_events.size(); j++) {
                    if (auto base_ev = dynamic_cast<ze_base_event*>(multiple_events->m_events[j].get())) {
                        auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                        if ((m_queue_stamp == 0) || (current_ev_queue_stamp > m_queue_stamp)) {
                            m_queue_stamp = current_ev_queue_stamp;
                            m_last_event = base_ev->get_handle();
                        }
                    }
                    m_events.push_back(multiple_events->m_events[j]);
                }
            } else {
                if (auto base_ev = dynamic_cast<ze_base_event*>(ev[i].get())) {
                    auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                    if ((m_queue_stamp == 0) || (current_ev_queue_stamp > m_queue_stamp)) {
                        m_queue_stamp = current_ev_queue_stamp;
                        m_last_event = base_ev->get_handle();
                    }
                }
                m_events.push_back(ev[i]);
            }
        }
    }

    ze_event_handle_t m_last_event = nullptr;
    std::vector<event::ptr> m_events;
    const ze_engine &m_engine;
};

}  // namespace ze
}  // namespace cldnn
