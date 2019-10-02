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
    if (!_events.empty()) {
        for (size_t i = 0; i < _events.size(); i++) {
            _events[i]->wait();
        }
    }
}

bool base_events::is_set_impl() {
    if (!_events.empty()) {
        for (size_t i = 0; i < _events.size(); i++) {
            if (!_events[i]->is_set())
                return false;
        }
        return true;
    }
    return true;
}

bool base_events::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    cl_ulong min_queue = CL_ULONG_MAX;
    cl_ulong min_sub = CL_ULONG_MAX;
    cl_ulong min_start = CL_ULONG_MAX;
    uint64_t execution_time = 0;

    for (size_t i = 0; i < _events.size(); i++) {
        auto be = dynamic_cast<base_event*>(_events[i].get());
        if (!is_event_profiled(be->_event))
            continue;

        cl_ulong curr_queue;
        cl_ulong curr_sub;
        cl_ulong curr_start;
        cl_ulong curr_end;
        be->_event.getProfilingInfo(CL_PROFILING_COMMAND_QUEUED, &curr_queue);
        be->_event.getProfilingInfo(CL_PROFILING_COMMAND_SUBMIT, &curr_sub);
        be->_event.getProfilingInfo(CL_PROFILING_COMMAND_START, &curr_start);
        be->_event.getProfilingInfo(CL_PROFILING_COMMAND_END, &curr_end);

        if (curr_queue < min_queue)
            min_queue = curr_queue;

        if (curr_sub < min_sub)
            min_sub = curr_sub;

        if (curr_start < min_start)
            min_start = curr_start;

        execution_time += curr_end - curr_start;
    }

    info.push_back(get_profiling_interval(profiling_periods[0].name, min_sub, min_queue));
    info.push_back(get_profiling_interval(profiling_periods[1].name, min_start, min_sub));
    info.push_back(get_profiling_interval(profiling_periods[2].name, 0, execution_time));

    return true;
}
