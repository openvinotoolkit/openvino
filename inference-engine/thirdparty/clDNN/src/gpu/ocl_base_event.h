/*
// Copyright (c) 2019 Intel Corporation
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
*/

#pragma once

#include "ocl_toolkit.h"
#include <vector>
#include <memory>
#include <list>

namespace cldnn {
namespace gpu {

struct profiling_period_ocl_start_stop {
    const char* name;
    cl_profiling_info start;
    cl_profiling_info stop;
};

struct ocl_base_event : virtual public event_impl {
public:
    explicit ocl_base_event(uint64_t queue_stamp = 0, bool valid = false) : _queue_stamp(queue_stamp) { _attached = valid; }
    uint64_t get_queue_stamp() const { return _queue_stamp; }
    virtual cl::Event get() = 0;

protected:
    uint64_t _queue_stamp = 0;
};

struct base_event : virtual public ocl_base_event {
public:
    base_event(std::shared_ptr<gpu_toolkit> ctx, cl::Event const& ev, uint64_t queue_stamp = 0)
        : ocl_base_event(queue_stamp, true), _ctx(ctx), _event(ev) {}

    explicit base_event(std::shared_ptr<gpu_toolkit> ctx) : ocl_base_event(0, false), _ctx(ctx) {}

    void attach_ocl_event(const cl::Event& ev, const uint64_t q_stamp) {
        _event = ev;
        _queue_stamp = q_stamp;
        _attached = true;
        _set = false;
    }

    std::shared_ptr<gpu_toolkit> get_context() const { return _ctx; }
    cl::Event get() override { return _event; }

private:
    std::shared_ptr<gpu_toolkit> _ctx;
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
    base_events(std::shared_ptr<gpu_toolkit> ctx, std::vector<event_impl::ptr> const& ev)
        : ocl_base_event(0, true), _ctx(ctx) {
        process_events(ev);
    }

    explicit base_events(std::shared_ptr<gpu_toolkit> ctx) : ocl_base_event(0, false), _ctx(ctx) {}

    void attach_events(const std::vector<event_impl::ptr>& ev) {
        if (_attached)
            throw std::runtime_error("Trying to attach events to valid event object.");
        process_events(ev);
        _attached = true;
    }

    cl::Event get() override { return _last_ocl_event; }
    std::shared_ptr<gpu_toolkit> get_context() const { return _ctx; }

    void reset() override {
        ocl_base_event::reset();
        _events.clear();
    }

private:
    void wait_impl() override;
    bool is_set_impl() override;

    void process_events(const std::vector<event_impl::ptr>& ev) {
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
    std::shared_ptr<gpu_toolkit> _ctx;
    std::vector<event_impl::ptr> _events;
};

}  // namespace gpu
}  // namespace cldnn
