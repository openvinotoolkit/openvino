// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "ocl_base_event.hpp"

#include <vector>
#include <memory>
#include <list>

namespace cldnn {
namespace ocl {

struct ocl_event : public ocl_base_event {
public:
    ocl_event(cl::Event const& ev, uint64_t queue_stamp = 0)
        : ocl_base_event(queue_stamp)
        , _event(ev) {}

    ocl_event(uint64_t duration_nsec, uint64_t queue_stamp = 0)
        : ocl_base_event(queue_stamp)
        , duration_nsec(duration_nsec) {}

    cl::Event& get() override { return _event; }

private:
    bool _callback_set = false;
    void set_ocl_callback();
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;
    bool add_event_handler_impl(event_handler, void*) override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    static void CL_CALLBACK ocl_event_completion_callback(cl_event, cl_int, void* me);

    friend struct ocl_events;

protected:
    cl::Event _event;
    std::optional<uint64_t> duration_nsec;
};

struct ocl_events : public ocl_base_event {
public:
    ocl_events(std::vector<event::ptr> const& ev)
        : ocl_base_event(0) {
        process_events(ev);
    }

    cl::Event& get() override { return _last_ocl_event; }

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
            auto multiple_events = dynamic_cast<ocl_events*>(ev[i].get());
            if (multiple_events) {
                for (size_t j = 0; j < multiple_events->_events.size(); j++) {
                    if (auto base_ev = dynamic_cast<ocl_event*>(multiple_events->_events[j].get())) {
                        auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                        if ((_queue_stamp == 0) || (current_ev_queue_stamp > _queue_stamp)) {
                            _queue_stamp = current_ev_queue_stamp;
                            _last_ocl_event = base_ev->get();
                        }
                    }
                    _events.push_back(multiple_events->_events[j]);
                }
            } else {
                if (auto base_ev = dynamic_cast<ocl_event*>(ev[i].get())) {
                    auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                    if ((_queue_stamp == 0) || (current_ev_queue_stamp > _queue_stamp)) {
                        _queue_stamp = current_ev_queue_stamp;
                        _last_ocl_event = base_ev->get();
                    }
                }
                _events.push_back(ev[i]);
            }
        }
    }

    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    cl::Event _last_ocl_event;
    std::vector<event::ptr> _events;
};

}  // namespace ocl
}  // namespace cldnn
