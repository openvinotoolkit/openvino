// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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

bool user_event::get_profiling_info_impl(std::list<cldnn::instrumentation::profiling_interval>& info) {
    if (_duration == nullptr) {
        return false;
    }

    auto period = std::make_shared<instrumentation::profiling_period_basic>(_duration->value());
    info.push_back({"duration", period });
    return true;
}
