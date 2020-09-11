/*
// Copyright (c) 2016 Intel Corporation
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

///////////////////////////////////////////////////////////////////////////////////////////////////
#pragma once
#include "api/event.hpp"
#include "refcounted_obj.h"

#include <list>
#include <mutex>
#include <utility>

namespace cldnn {
struct user_event;

struct event_impl : public refcounted_obj<event_impl> {
public:
    event_impl() = default;

    void wait();
    bool is_set();
    virtual bool is_valid() const { return _attached; }
    virtual void reset() {
        _attached = false;
        _set = false;
        _profiling_captured = false;
        _profiling_info.clear();
    }
    // returns true if handler has been successfully added
    bool add_event_handler(event_handler handler, void* data);

    const std::list<instrumentation::profiling_interval>& get_profiling_info();

private:
    std::mutex _handlers_mutex;
    std::list<std::pair<event_handler, void*>> _handlers;

    bool _profiling_captured = false;
    std::list<instrumentation::profiling_interval> _profiling_info;

protected:
    bool _set = false;
    bool _attached =
        false;  // because ocl event can be attached later, we need mechanism to check if such event was attached
    void call_handlers();

    virtual void wait_impl() = 0;
    virtual bool is_set_impl() = 0;
    virtual bool add_event_handler_impl(event_handler, void*) { return true; }

    // returns whether profiling info has been captures successfully and there's no need to call this impl a second time
    // when user requests to get profling info
    virtual bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>&) { return true; }
};

struct user_event : virtual public event_impl {
public:
    explicit user_event(bool set = false) { _set = set; }

    void set() {
        if (_set)
            return;
        _set = true;
        set_impl();
        call_handlers();
    }

private:
    virtual void set_impl() = 0;
};

}  // namespace cldnn
