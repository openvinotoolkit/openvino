// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "sycl_event.hpp"

#include <cassert>
#include <iostream>
#include <vector>
#include <list>
#include <map>

namespace cldnn {
namespace sycl {

void sycl_event::wait_impl() {
    if (_event.get() != nullptr) {
        _event.wait();
    }
}

bool sycl_event::is_set_impl() {
    return _event.get_info<cl::sycl::info::event::command_execution_status>() == cl::sycl::info::event_command_status::complete;
}

bool sycl_event::get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) {
    return true;
}

} // namespace sycl
} // namespace cldnn
