// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "sycl_common.hpp"
#include "cldnn/runtime/event.hpp"

#include <vector>
#include <memory>
#include <list>

namespace cldnn {
namespace sycl {

struct sycl_event : public event {
public:
    sycl_event(const cl::sycl::context& /* ctx */, cl::sycl::event const& ev, uint64_t queue_stamp = 0)
        :  _event(ev), _queue_stamp(queue_stamp) { _attached = true; }

    sycl_event(const cl::sycl::context& /* ctx */) {}

    void attach_ocl_event(const cl::sycl::event& ev, const uint64_t q_stamp) {
        _event = ev;
        _queue_stamp = q_stamp;
        _attached = true;
        _set = false;
    }

    cl::sycl::event get() { return _event; }

    uint64_t get_queue_stamp() const { return _queue_stamp; }

private:
    void wait_impl() override;
    bool is_set_impl() override;
    bool add_event_handler_impl(event_handler, void*) override { return false; }
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;

protected:
    cl::sycl::event _event;
    uint64_t _queue_stamp = 0;
};

}  // namespace sycl
}  // namespace cldnn
