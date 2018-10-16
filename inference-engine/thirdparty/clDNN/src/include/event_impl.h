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
#include "api_impl.h"
#include "refcounted_obj.h"

#include <list>
#include <mutex>

namespace cldnn
{
struct user_event;

struct event_impl : public refcounted_obj<event_impl>
{
public:
    event_impl() = default;

    void wait();
    bool is_set();
    
    //returns true if handler has been successfully added
    bool add_event_handler(cldnn_event_handler handler, void* data);
    
    const std::list<cldnn_profiling_interval>& get_profiling_info();

private:
    std::mutex _handlers_mutex;
    std::list<std::pair<cldnn_event_handler, void*>> _handlers;

    bool _profiling_captured = false;
    std::list<cldnn_profiling_interval> _profiling_info;

protected:
    bool _set = false;

    void call_handlers();

    virtual void wait_impl() = 0;
    virtual bool is_set_impl() = 0;
    virtual bool add_event_handler_impl(cldnn_event_handler, void*) { return true; }

    //returns whether profiling info has been captures successfully and there's no need to call this impl a second time when user requests to get profling info
    virtual bool get_profiling_info_impl(std::list<cldnn_profiling_interval>&) { return true; };
};

struct user_event : virtual public event_impl
{
public:
    user_event(bool set = false)
    {
        _set = set;
    }

    void set()
    { 
        if (_set)
            return;
        _set = true;
        set_impl();
        call_handlers();
    }

private:
    virtual void set_impl() = 0;
};

}

API_CAST(::cldnn_event, cldnn::event_impl)
