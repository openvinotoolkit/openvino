// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sycl_common.hpp"
#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/utils.hpp"

#include <vector>
#include <memory>
#include <list>

namespace cldnn {
namespace sycl {

struct sycl_base_event : public event {
public:
    explicit sycl_base_event(uint64_t queue_stamp = 0) : event(), _queue_stamp(queue_stamp) { }
    uint64_t get_queue_stamp() const { return _queue_stamp; }
    virtual ::sycl::event& get() = 0;

protected:
    uint64_t _queue_stamp = 0;
};

}  // namespace sycl
}  // namespace cldnn
