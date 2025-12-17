// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sycl_common.hpp"
#include "sycl_base_event.hpp"

#include <vector>
#include <memory>
#include <list>

namespace cldnn {
namespace sycl {
namespace utils {
std::vector<::sycl::event> get_sycl_events(const std::vector<event::ptr>& events);
}  // namespace utils

struct sycl_event : public sycl_base_event {
public:
    sycl_event(::sycl::event const& ev, ::sycl::queue queue, uint64_t queue_stamp = 0)
        : sycl_base_event(queue_stamp)
        , _event(ev)
        , _queue(queue) {}

    sycl_event(uint64_t duration_nsec, uint64_t queue_stamp = 0)
        : sycl_base_event(queue_stamp)
        , duration_nsec(duration_nsec) {}

    ::sycl::event& get() override { return _event; }

private:
    bool _callback_set = false;
    void set_sycl_callback();
    void wait_impl() override;
    void set_impl() override;
    bool is_set_impl() override;
    bool add_event_handler_impl(event_handler, void*) override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    friend struct sycl_events;

protected:
    ::sycl::event _event;
    ::sycl::queue _queue;
    std::optional<uint64_t> duration_nsec;
};

struct sycl_events : public sycl_base_event {
public:
    sycl_events(std::vector<event::ptr> const& ev)
        : sycl_base_event(0) {
            process_events(ev);
        }

    ::sycl::event& get() override { return _last_sycl_event; }

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
            auto multiple_events = dynamic_cast<sycl_events*>(ev[i].get());
            if (multiple_events) {
                for (size_t j = 0; j < multiple_events->_events.size(); j++) {
                    if (auto base_ev = dynamic_cast<sycl_event*>(multiple_events->_events[j].get())) {
                        auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                        if ((_queue_stamp == 0) || (current_ev_queue_stamp > _queue_stamp)) {
                            _queue_stamp = current_ev_queue_stamp;
                            _last_sycl_event = base_ev->get();
                        }
                    }
                    _events.push_back(multiple_events->_events[j]);
                }
            } else {
                if (auto base_ev = dynamic_cast<sycl_event*>(ev[i].get())) {
                    auto current_ev_queue_stamp = base_ev->get_queue_stamp();
                    if ((_queue_stamp == 0) || (current_ev_queue_stamp > _queue_stamp)) {
                        _queue_stamp = current_ev_queue_stamp;
                        _last_sycl_event = base_ev->get();
                    }
                }
                _events.push_back(ev[i]);
            }
        }
    }

    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

    ::sycl::event _last_sycl_event;
    std::vector<event::ptr> _events;
};

}  // namespace sycl
}  // namespace cldnn
