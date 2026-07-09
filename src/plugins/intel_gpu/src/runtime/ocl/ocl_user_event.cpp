// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "ocl_user_event.hpp"
#include <list>

using namespace cldnn::ocl;

ocl_user_event::ocl_user_event(const cl::Context& ctx,
                               bool is_set,
                               bool has_device_steady_offset,
                               std::chrono::nanoseconds device_steady_offset)
    : ocl_base_event()
    , _ctx(ctx)
    , _event(_ctx) {
    if (has_device_steady_offset) {
        auto steady_now = std::chrono::duration_cast<std::chrono::nanoseconds>(std::chrono::steady_clock::now().time_since_epoch());
        auto device_time = steady_now + device_steady_offset;
        _exec_start = device_time > std::chrono::nanoseconds::zero() ? device_time : std::chrono::nanoseconds::zero();
        _has_device_timestamps = true;
    }

    if (is_set) {
        set();
    }
}

void ocl_user_event::set_impl() {
    // we simulate "wrapper_cast" here to cast from cl::Event to cl::UserEvent which both wrap the same cl_event
    // casting is valid as long as cl::UserEvent does not add any members to cl::Event (which it shouldn't)
    static_assert(sizeof(cl::UserEvent) == sizeof(cl::Event) && alignof(cl::UserEvent) == alignof(cl::Event),
                  "cl::UserEvent does not match cl::Event");
    static_cast<cl::UserEvent&&>(get()).setStatus(CL_COMPLETE);
    _duration = std::unique_ptr<cldnn::instrumentation::profiling_period_basic>(
        new cldnn::instrumentation::profiling_period_basic(_timer.uptime()));
}

bool ocl_user_event::get_profiling_info_impl(std::list<cldnn::instrumentation::profiling_interval>& info) {
    if (_duration == nullptr) {
        return false;
    }

    auto period = std::make_shared<instrumentation::profiling_period_basic>(_duration->value());
    info.push_back({ instrumentation::profiling_stage::duration, period });

    if (_has_device_timestamps) {
        auto zero_period = std::make_shared<instrumentation::profiling_period_basic>(std::chrono::nanoseconds::zero());
        info.push_back({ instrumentation::profiling_stage::starting, zero_period, _exec_start, true });
        info.push_back({ instrumentation::profiling_stage::executing, period, _exec_start, true });
    } else {
        info.push_back({ instrumentation::profiling_stage::executing, period });
    }

    return true;
}

void ocl_user_event::wait_impl() {
    if (!_set) {
        throw std::runtime_error("[CLDNN] ocl_user_event::wait_impl is called before marking event handle as complete");
    }

    if (_event.get() != nullptr) {
        try {
            _event.wait();
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    }
}

bool ocl_user_event::is_set_impl() {
    if (_event.get() != nullptr) {
        try {
            return _event.getInfo<CL_EVENT_COMMAND_EXECUTION_STATUS>() == CL_COMPLETE;
        } catch (cl::Error const& err) {
            OPENVINO_THROW(OCL_ERR_MSG_FMT(err));
        }
    }
    return true;
}
