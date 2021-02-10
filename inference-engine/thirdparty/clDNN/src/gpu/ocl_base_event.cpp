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

#include "ocl_base_event.h"

#include <cassert>
#include <iostream>
#include <vector>
#include <list>
using namespace cldnn;
using namespace gpu;

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

instrumentation::profiling_interval get_profiling_interval(const char* name, cl_ulong start,  cl_ulong end) {
    auto diff = std::chrono::nanoseconds(end - start);
    auto period = std::make_shared<instrumentation::profiling_period_basic>(diff);
    return { name, period };
}

}  // namespace

void CL_CALLBACK base_event::ocl_event_completion_callback(cl_event, cl_int, void* me) {
    reinterpret_cast<base_event*>(me)->_set = true;
    reinterpret_cast<base_event*>(me)->call_handlers();
}

void base_event::set_ocl_callback() {
    if (_callback_set)
        return;

    if (_event.get() != nullptr) {
        _event.setCallback(CL_COMPLETE, ocl_event_completion_callback, this);
        _callback_set = true;
    }
}

void base_event::wait_impl() {
    if (_event.get() != nullptr) {
        _event.wait();
        if (get_context()->logging_enabled()) {
            get_context()->log(0, "Wait for event: " + std::to_string(_queue_stamp));
        }
    }
}

bool base_event::is_set_impl() {
    if (_event.get() != nullptr) {
        return _event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE;
    }
    return true;
}

bool base_event::add_event_handler_impl(event_handler, void*) {
    set_ocl_callback();
    return true;
}

static const std::vector<profiling_period_ocl_start_stop> profiling_periods{
    {"submission", CL_PROFILING_COMMAND_QUEUED, CL_PROFILING_COMMAND_SUBMIT},
    {"starting", CL_PROFILING_COMMAND_SUBMIT, CL_PROFILING_COMMAND_START},
    {"executing", CL_PROFILING_COMMAND_START, CL_PROFILING_COMMAND_END},
};

bool base_event::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    if (!is_event_profiled(_event))
        return true;

    for (auto& period : profiling_periods) {
        cl_ulong start;
        cl_ulong end;

        _event.getProfilingInfo(period.start, &start);
        _event.getProfilingInfo(period.stop, &end);

        info.push_back(get_profiling_interval(period.name, start, end));
    }

    return true;
}

void base_events::wait_impl() {
    if (_last_ocl_event.get() != nullptr) {
        _last_ocl_event.wait();
        if (get_context()->logging_enabled()) {
            get_context()->log(0, "Wait for event: " + std::to_string(_queue_stamp));
        }
    }
}

bool base_events::is_set_impl() {
    if (_last_ocl_event.get() != nullptr) {
        return _last_ocl_event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE;
    }
    return true;
}

bool base_events::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {

    // For every profiling period (i.e. submission / starting / executing),
    // the goal is to sum up all disjoint durations of its projection on the time axis

    std::map<std::string, std::vector<std::pair<unsigned long long, unsigned long long>>> all_durations;

    for (size_t i = 0; i < _events.size(); i++) {
        auto be = dynamic_cast<base_event*>(_events[i].get());
        if (!is_event_profiled(be->_event))
            continue;

        for (auto& period : profiling_periods) {
            cl_ulong ev_start;
            cl_ulong ev_end;
            be->_event.getProfilingInfo(period.start, &ev_start);
            be->_event.getProfilingInfo(period.stop, &ev_end);
            auto ev_duration = std::make_pair(static_cast<unsigned long long>(ev_start),
                                              static_cast<unsigned long long>(ev_end));

            auto& durations = all_durations[period.name];
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
        for (auto& duration : all_durations[period.name]) {
            sum += (duration.second - duration.first);
        }
        info.push_back(get_profiling_interval(period.name, 0, sum));
    }

    return true;
}
