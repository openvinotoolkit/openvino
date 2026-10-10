// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#include "sycl_user_event.hpp"

#include <list>
#include <memory>

namespace cldnn {
namespace sycl {

void sycl_user_event::set_impl() {
    // There is nothing to signal on the SYCL side: cldnn::event::set() has already updated the host
    // side state, which is the only representation of the completion of this event.
    _duration = std::make_unique<instrumentation::profiling_period_basic>(_timer.uptime());
}

bool sycl_user_event::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    if (_duration == nullptr) {
        return false;
    }

    auto period = std::make_shared<instrumentation::profiling_period_basic>(_duration->value());
    info.push_back({ instrumentation::profiling_stage::duration, period });
    info.push_back({ instrumentation::profiling_stage::executing, period });
    return true;
}

void sycl_user_event::wait_impl() {
    // cldnn::event::wait() returns early when the event is already set, so reaching this point means
    // somebody is waiting for a user event which will never be signaled.
    OPENVINO_THROW("[GPU] sycl_user_event::wait_impl is called before marking event handle as complete");
}

bool sycl_user_event::is_set_impl() {
    // The host side cldnn::event::_set flag is the only source of truth for this event, and
    // cldnn::event::is_set() has already checked it before calling this. No other place holds the
    // completion state, so the event is known to be not set here.
    return false;
}

}  // namespace sycl
}  // namespace cldnn
