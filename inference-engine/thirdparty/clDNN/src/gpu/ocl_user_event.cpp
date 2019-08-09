/*
// Copyright (c) 2018 Intel Corporation
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

#include "ocl_user_event.h"
#include <list>

using namespace cldnn::gpu;

void user_event::set_impl() {
    // we simulate "wrapper_cast" here to cast from cl::Event to cl::UserEvent which both wrap the same cl_event
    // casting is valid as long as cl::UserEvent does not add any members to cl::Event (which it shouldn't)
    static_assert(sizeof(cl::UserEvent) == sizeof(cl::Event) && alignof(cl::UserEvent) == alignof(cl::Event),
                  "cl::UserEvent does not match cl::Event");
    static_cast<cl::UserEvent&&>(get()).setStatus(CL_COMPLETE);
    _duration = std::unique_ptr<cldnn::instrumentation::profiling_period_basic>(
        new cldnn::instrumentation::profiling_period_basic(_timer.uptime()));
    _attached = true;
}

bool user_event::get_profiling_info_impl(std::list<cldnn_profiling_interval>& info) {
    if (_duration == nullptr) {
        return false;
    }

    info.push_back({"duration", static_cast<uint64_t>(_duration->value().count())});
    return true;
}