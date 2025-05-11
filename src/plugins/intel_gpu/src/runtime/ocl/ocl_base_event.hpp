// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "ocl_common.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/utils.hpp"

#include <vector>
#include <memory>
#include <list>

namespace cldnn {
namespace ocl {

struct profiling_period_ocl_start_stop {
    instrumentation::profiling_stage stage;
    cl_profiling_info start;
    cl_profiling_info stop;
};

struct ocl_base_event : public event {
public:
    explicit ocl_base_event(uint64_t queue_stamp = 0) : event(), _queue_stamp(queue_stamp) { }
    uint64_t get_queue_stamp() const { return _queue_stamp; }
    virtual cl::Event& get() = 0;

protected:
    uint64_t _queue_stamp = 0;
};

}  // namespace ocl
}  // namespace cldnn
