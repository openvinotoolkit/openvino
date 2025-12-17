// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_event.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <list>
#include <map>

using namespace cldnn;
using namespace cldnn::sycl;

namespace {
bool is_event_profiled(const ::sycl::queue& queue) {
    return queue.has_property<::sycl::property::queue::enable_profiling>();
}

instrumentation::profiling_interval get_profiling_interval(instrumentation::profiling_stage stage, uint64_t start,  uint64_t end) {
    auto diff = std::chrono::nanoseconds(end - start);
    auto period = std::make_shared<instrumentation::profiling_period_basic>(diff);
    return { stage, period };
}

const std::pair<uint64_t, uint64_t> get_profiling_period_value(const ::sycl::event& event, instrumentation::profiling_stage stage) {
    switch (stage) {
        case instrumentation::profiling_stage::starting:
            return std::make_pair(event.get_profiling_info<::sycl::info::event_profiling::command_submit>(),
                                  event.get_profiling_info<::sycl::info::event_profiling::command_start>());
        case instrumentation::profiling_stage::executing:
            return std::make_pair(event.get_profiling_info<::sycl::info::event_profiling::command_start>(),
                                  event.get_profiling_info<::sycl::info::event_profiling::command_end>());
        default:
            OPENVINO_THROW("Unsupported profiling stage for SYCL event: ", static_cast<int>(stage));
    }
}

}  // namespace

namespace cldnn::sycl::utils {
std::vector<::sycl::event> get_sycl_events(const std::vector<event::ptr>& events) {
    std::vector<::sycl::event> sycl_events;
    for (const auto& ev : events) {
        if (auto sycl_base_ev = dynamic_cast<sycl_base_event*>(ev.get())) {
            sycl_events.push_back(sycl_base_ev->get());
        }
    }

    return sycl_events;
}
}  // namespace cldnn::sycl::utils

void sycl_event::set_sycl_callback() {
    if (_callback_set)
        return;

    _queue.submit([this](::sycl::handler& cgh) {
        cgh.depends_on(_event);
        cgh.host_task([this]() {
            this->_set = true;
            this->call_handlers();
        });
    });
}

void sycl_event::wait_impl() {
    try {
        GPU_DEBUG_TRACE_DETAIL << "sycl_event::wait_impl: waiting for event: " << &_event << std::endl;
        _event.wait_and_throw();
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

void sycl_event::set_impl() {
    wait_impl();
}

bool sycl_event::is_set_impl() {
    try {
        return _event.get_info<::sycl::info::event::command_execution_status>() == ::sycl::info::event_command_status::complete;
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
    return true;
}

bool sycl_event::add_event_handler_impl(event_handler, void*) {
    set_sycl_callback();
    return true;
}

static const std::vector<instrumentation::profiling_stage> profiling_stages {
    // There is no CL_PROFILING_COMMAND_QUEUED in SYCL, so we skip it
    // instrumentation::profiling_stage::submission,
    instrumentation::profiling_stage::starting,
    instrumentation::profiling_stage::executing,
};


bool sycl_event::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    if (duration_nsec.has_value()) {
        auto stage = instrumentation::profiling_stage::executing;
        auto duration = std::chrono::nanoseconds(duration_nsec.value());
        auto period = std::make_shared<instrumentation::profiling_period_basic>(duration);

        info.push_back({ stage, period });

        return true;
    }

    if (!is_event_profiled(_queue))
        return true;

    for (auto& stage : profiling_stages) {
        uint64_t start;
        uint64_t end;

        try {
            std::tie(start, end) = get_profiling_period_value(_event, stage);
        } catch (::sycl::exception const& err) {
            OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
        }

        info.push_back(get_profiling_interval(stage, start, end));
    }

    return true;
}

void sycl_events::wait_impl() {
    try {
        _last_sycl_event.wait();
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
}

void sycl_events::set_impl() {
    wait_impl();
}

bool sycl_events::is_set_impl() {
    try {
        return _last_sycl_event.get_info<::sycl::info::event::command_execution_status>() == ::sycl::info::event_command_status::complete;
    } catch (::sycl::exception const& err) {
        OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
    }
    return true;
}

bool sycl_events::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    // For every profiling period (i.e. submission / starting / executing),
    // the goal is to sum up all disjoint durations of its projection on the time axis

    std::map<instrumentation::profiling_stage, std::vector<std::pair<uint64_t, uint64_t>>> all_durations;

    for (size_t i = 0; i < _events.size(); i++) {
        sycl_event *be = nullptr;

        try {
            be = downcast<sycl_event>(_events[i].get());
        } catch (const ov::Exception &err) {
            GPU_DEBUG_LOG << "WARNING: failed to downcast event to sycl_event - " << err.what() << std::endl;
            continue;
        }

        if (!is_event_profiled(be->_queue))
            continue;

        for (auto& stage : profiling_stages) {
            std::pair<uint64_t, uint64_t> ev_duration;
            try {
                ev_duration = get_profiling_period_value(be->_event, stage);
            } catch (::sycl::exception const& err) {
                OPENVINO_THROW(SYCL_ERR_MSG_FMT(err));
            }
            auto& durations = all_durations[stage];
            bool ev_duration_merged = false;
            auto it = durations.begin();

            while (it != durations.end()) {
                auto& duration = *it;
                if ((duration.second >= ev_duration.first) && (duration.first <= ev_duration.second)) {
                    if ((duration.first == ev_duration.first) && (duration.second == ev_duration.second)) {
                        if (!ev_duration_merged) {
                            ev_duration_merged = true;
                            break;
                        } else {
                            it = durations.erase(it);
                        }
                    } else {
                        if (!ev_duration_merged) {
                            duration.first = std::min(duration.first, ev_duration.first);
                            duration.second = std::max(duration.second, ev_duration.second);
                            ev_duration = duration;
                            ev_duration_merged = true;
                            it++;
                        } else {
                            if (duration.second > ev_duration.second) {
                                ev_duration.second = duration.second;
                                it--;
                                it->second = ev_duration.second;
                                it++;
                            }
                            it = durations.erase(it);
                        }
                    }
                } else {
                    it++;
                }
            }

            if (!ev_duration_merged) {
                durations.insert(it, ev_duration);
            }
        }
    }

    for (auto& stage : profiling_stages) {
        uint64_t sum = 0;
        for (auto& duration : all_durations[stage]) {
            sum += (duration.second - duration.first);
        }

        info.push_back(get_profiling_interval(stage, 0, sum));
    }

    return true;
}
