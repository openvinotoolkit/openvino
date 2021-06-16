// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "cldnn/runtime/event.hpp"
#include "cldnn/runtime/utils.hpp"

#include <vector>
#include <memory>
#include <list>

namespace cldnn {
namespace ocl {

struct profiling_period_ocl_start_stop {
    const char* name;
    cl_profiling_info start;
    cl_profiling_info stop;
};

struct ocl_base_event : virtual public event {
public:
    explicit ocl_base_event(uint64_t queue_stamp = 0, bool valid = false) : _queue_stamp(queue_stamp) { _attached = valid; }
    uint64_t get_queue_stamp() const { return _queue_stamp; }
    virtual cl::Event get() = 0;

protected:
    uint64_t _queue_stamp = 0;
};

struct base_event : virtual public ocl_base_event {
public:
    base_event(const cl::Context& /* ctx */, cl::Event const& ev, uint64_t queue_stamp = 0)
        : ocl_base_event(queue_stamp, true), _event(ev) {}

    base_event(const cl::Context& /* ctx */) : ocl_base_event(0, false) {}

    void attach_ocl_event(const cl::Event& ev, const uint64_t q_stamp) {
        _event = ev;
        _queue_stamp = q_stamp;
        _attached = true;
        _set = false;
    }

    cl::Event get() override { return _event; }

private:
    bool _callback_set = false;
    void set_ocl_callback();
    static void CL_CALLBACK ocl_event_completion_callback(cl_event, cl_int, void* me);

private:
    void wait_impl() override;
    bool is_set_impl() override;
    bool add_event_handler_impl(event_handler, void*) override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    friend struct base_events;

protected:
    cl::Event _event;
};

struct base_events : virtual public ocl_base_event {
public:
    base_events(const cl::Context& /* ctx */, std::vector<event::ptr> const& ev)
        : ocl_base_event(0, true) {
        process_events(ev);
    }

    base_events(const cl::Context& /* ctx */) : ocl_base_event(0, false) {}

    void attach_events(const std::vector<event::ptr>& ev) {
        if (_attached)
            throw std::runtime_error("Trying to attach events to valid event object.");
        process_events(ev);
        _attached = true;
    }

    cl::Event get() override { return _last_ocl_event; }

    void reset() override {
        ocl_base_event::reset();
        _events.clear();
    }

private:
    void wait_impl() override;
    bool is_set_impl() override;

    void process_events(const std::vector<event::ptr>& ev) {
        for (size_t i = 0; i < ev.size(); i++) {
            auto multiple_events = dynamic_cast<base_events*>(ev[i].get());
            if (multiple_events) {
                for (size_t j = 0; j < multiple_events->_events.size(); j++) {
                    if (auto base_ev = dynamic_cast<base_event*>(multiple_events->_events[j].get())) {
                        auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                        if ((_queue_stamp == 0) || (current_ev_queue_stamp > _queue_stamp)) {
                            _queue_stamp = current_ev_queue_stamp;
                            _last_ocl_event = base_ev->get();
                        }
                    }
                    _events.push_back(multiple_events->_events[j]);
                }
            } else {
                if (auto base_ev = dynamic_cast<base_event*>(ev[i].get())) {
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
