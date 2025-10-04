// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_base_event.hpp"
#include "ze_event_manager.hpp"
#include "ze_event_pool.hpp"

#include <vector>
#include <list>

namespace cldnn {
namespace ze {

struct ze_event : public ze_base_event {
public:
    ze_event(ze_event_manager *ev_manager, ze_event_handle_t ev, uint64_t queue_stamp = 0, std::shared_ptr<ze_event_pool> event_pool = nullptr)
        : ze_base_event(queue_stamp)
        , m_event_manager(ev_manager)
        , m_event_pool(event_pool)
        , m_event(ev) {}

    ze_event_handle_t get() override { return m_event; }
    bool is_profiled() const;

    ~ze_event();

private:
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    friend struct ze_events;

protected:
    ze_event_manager *m_event_manager;
    std::shared_ptr<ze_event_pool> m_event_pool = nullptr;
    ze_event_handle_t m_event;
};

struct ze_events : public ze_base_event {
public:
    ze_events(std::vector<event::ptr> const& ev)
        : ze_base_event(0) {
        process_events(ev);
    }

    ze_event_handle_t get() override { return _last_ze_event; }

    void reset() override {
        event::reset();
        _events.clear();
    }

private:
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;

    void process_events(const std::vector<event::ptr>& ev) {
        for (size_t i = 0; i < ev.size(); i++) {
            auto multiple_events = dynamic_cast<ze_events*>(ev[i].get());
            if (multiple_events) {
                for (size_t j = 0; j < multiple_events->_events.size(); j++) {
                    if (auto base_ev = dynamic_cast<ze_event*>(multiple_events->_events[j].get())) {
                        auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                        if ((_queue_stamp == 0) || (current_ev_queue_stamp > _queue_stamp)) {
                            _queue_stamp = current_ev_queue_stamp;
                            _last_ze_event = base_ev->get();
                        }
                    }
                    _events.push_back(multiple_events->_events[j]);
                }
            } else {
                if (auto base_ev = dynamic_cast<ze_event*>(ev[i].get())) {
                    auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                    if ((_queue_stamp == 0) || (current_ev_queue_stamp > _queue_stamp)) {
                        _queue_stamp = current_ev_queue_stamp;
                        _last_ze_event = base_ev->get();
                    }
                }
                _events.push_back(ev[i]);
            }
        }
    }

    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    ze_event_handle_t _last_ze_event;
    std::vector<event::ptr> _events;
};

}  // namespace ze
}  // namespace cldnn
