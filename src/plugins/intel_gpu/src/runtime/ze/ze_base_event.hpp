// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ze_common.hpp"
#include "cldnn/runtime/event.hpp"
#include "cldnn/runtime/utils.hpp"

#include <vector>
#include <memory>
#include <list>

namespace cldnn {
namespace ze {

// struct profiling_period_ze_start_stop {
//     const char* name;
//     cl_profiling_info start;
//     cl_profiling_info stop;
// };

struct ze_base_event : public event {
public:
    explicit ze_base_event(uint64_t queue_stamp = 0) : event(), _queue_stamp(queue_stamp) { }
    uint64_t get_queue_stamp() const { return _queue_stamp; }
    virtual ze_event_handle_t get() = 0;

protected:
    uint64_t _queue_stamp = 0;
};

}  // namespace ze
}  // namespace cldnn
