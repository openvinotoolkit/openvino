// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_event.hpp"
#include "intel_gpu/runtime/debug_configuration.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <list>
#include <map>

using namespace cldnn;
using namespace ocl;

namespace {
bool is_event_profiled(const cl::Event& event) {
    if (event() != nullptr) {
        auto queue = event.getInfo<CL_EVENT_COMMAND_QUEUE>();
        if (queue() != nullptr) {
            return (queue.getInfo<CL_QUEUE_PROPERTIES>() & CL_QUEUE_PROFILING_ENABLE) != 0;
        }
    }
    return false;
}

instrumentation::profiling_interval get_profiling_interval(instrumentation::profiling_stage stage, cl_ulong start,  cl_ulong end) {
    auto diff = std::chrono::nanoseconds(end - start);
    auto period = std::make_shared<instrumentation::profiling_period_basic>(diff);
    return { stage, period };
}

}  // namespace

void CL_CALLBACK ocl_event::ocl_event_completion_callback(cl_event, cl_int, void* me) {
    reinterpret_cast<ocl_event*>(me)->_set = true;
    reinterpret_cast<ocl_event*>(me)->call_handlers();
}

void ocl_event::set_ocl_callback() {
    if (_callback_set)
        return;

    if (_event.get() != nullptr) {
        _event.setCallback(CL_COMPLETE, ocl_event_completion_callback, this);
        _callback_set = true;
    }
}

void ocl_event::wait_impl() {
    if (_event.get() != nullptr) {
        try {
            _event.wait();
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    }
}

void ocl_event::set_impl() {
    wait_impl();
}

bool ocl_event::is_set_impl() {
    try {
        if (_event.get() != nullptr) {
            return _event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE;
        }
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
    return true;
}

bool ocl_event::add_event_handler_impl(event_handler, void*) {
    set_ocl_callback();
    return true;
}

static const std::vector<profiling_period_ocl_start_stop> profiling_periods{
    { instrumentation::profiling_stage::submission, CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_SUBMIT },
    { instrumentation::profiling_stage::starting, CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_START },
    { instrumentation::profiling_stage::executing, CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END },
};

bool ocl_event::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    if (duration_nsec.has_value()) {
        auto stage = instrumentation::profiling_stage::executing;
        auto duration = std::chrono::nanoseconds(duration_nsec.value());
        auto period = std::make_shared<instrumentation::profiling_period_basic>(duration);

        info.push_back({ stage, period });

        return true;
    }

    if (!is_event_profiled(_event))
        return true;

    for (auto& period : profiling_periods) {
        cl_ulong start;
        cl_ulong end;

        try {
            _event.getProfilingInfo(period.start, &start);
            _event.getProfilingInfo(period.stop, &end);
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }

        info.push_back(get_profiling_interval(period.stage, start, end));
    }

    return true;
}

void ocl_events::wait_impl() {
    if (_last_ocl_event.get() != nullptr) {
        try {
            _last_ocl_event.wait();
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    }
}

void ocl_events::set_impl() {
    wait_impl();
}

bool ocl_events::is_set_impl() {
    try {
        if (_last_ocl_event.get() != nullptr) {
            return _last_ocl_event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE;
        }
    } catch (cl::Error const& err) {
        OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
    }
    return true;
}

bool ocl_events::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    // For every profiling period (i.e. submission / starting / executing),
    // the goal is to sum up all disjoint durations of its projection on the time axis

    std::map<instrumentation::profiling_stage, std::vector<std::pair<unsigned long long, unsigned long long>>> all_durations;

    for (size_t i = 0; i < _events.size(); i++) {
        ocl_event *be = nullptr;

        try {
            be = downcast<ocl_event>(_events[i].get());
        } catch (const ov::Exception &err) {
            GPU_DEBUG_LOG << "WARNING: failed to downcast event to ocl_event - " << err.what() << std::endl;
            continue;
        }

        if (!is_event_profiled(be->_event))
            continue;

        for (auto& period : profiling_periods) {
            cl_ulong ev_start;
            cl_ulong ev_end;
            try {
                be->_event.getProfilingInfo(period.start, &ev_start);
                be->_event.getProfilingInfo(period.stop, &ev_end);
            } catch (cl::Error const& err) {
                OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
            }
            auto ev_duration = std::make_pair(static_cast<unsigned long long>(ev_start),
                                              static_cast<unsigned long long>(ev_end));

            auto& durations = all_durations[period.stage];
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

    for (auto& period : profiling_periods) {
        unsigned long long sum = 0;
        for (auto& duration : all_durations[period.stage]) {
            sum += (duration.second - duration.first);
        }

        info.push_back(get_profiling_interval(period.stage, 0, sum));
    }

    return true;
}
