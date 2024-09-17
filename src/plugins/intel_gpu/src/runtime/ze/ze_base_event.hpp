// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include <level_zero/ze_api.h>

namespace cldnn {
namespace ze {

struct ze_base_event : public event {
public:
    explicit ze_base_event(uint64_t queue_stamp = 0) : event(), _queue_stamp(queue_stamp) { }
    uint64_t get_queue_stamp() const { return _queue_stamp; }
    void set_queue_stamp(uint64_t val) { _queue_stamp = val; }
    virtual ze_event_handle_t get() = 0;

protected:
    uint64_t _queue_stamp = 0;
};

}  // namespace ze
}  // namespace cldnn
