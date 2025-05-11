// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/engine.hpp"

#include <list>
#include <vector>
#include <algorithm>

namespace cldnn {

void event::wait() {
    if (_set)
        return;

    // TODO: refactor in context of multiple simultaneous calls (for generic engine)
    wait_impl();
    _set = true;
    return;
}

void event::set() {
    if (_set)
        return;
    _set = true;
    set_impl();
    call_handlers();
}

bool event::is_set() {
    if (_set)
        return true;

    // TODO: refactor in context of multiple simultaneous calls (for generic engine)
    _set = is_set_impl();
    return _set;
}

bool event::add_event_handler(event_handler handler, void* data) {
    if (is_set()) {
        handler(data);
        return true;
    }

    std::lock_guard<std::mutex> lock(_handlers_mutex);
    auto itr = _handlers.insert(_handlers.end(), {handler, data});
    auto ret = add_event_handler_impl(handler, data);
    if (!ret)
        _handlers.erase(itr);

    return ret;
}

std::vector<instrumentation::profiling_interval> event::get_profiling_info() {
    if (!_profiling_captured) {
        _profiling_captured = get_profiling_info_impl(_profiling_info);
    }

    std::vector<instrumentation::profiling_interval> result(_profiling_info.size());
    std::copy(_profiling_info.begin(), _profiling_info.end(), result.begin());
    return result;
}

void event::call_handlers() {
    std::lock_guard<std::mutex> lock(_handlers_mutex);
    for (auto& pair : _handlers) {
        try {
            pair.first(pair.second);
        } catch (...) {
        }
    }
    _handlers.clear();
}

}  // namespace cldnn
