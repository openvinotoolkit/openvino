// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "intel_gpu/runtime/profiling.hpp"
#include "sycl_base_event.hpp"
#include <memory>
#include <list>

namespace cldnn {
namespace sycl {

// A user event, i.e. an event which is not backed by any command enqueued to a SYCL queue and whose
// completion state is controlled by the host via set().
//
// SYCL provides no way to express such an event: a default constructed ::sycl::event is specified to
// be always in the "complete" state and there is no API to signal it later (unlike cl::UserEvent or
// zeEventHostSignal). Therefore the completion state is tracked purely on the host side by
// cldnn::event::_set and the underlying ::sycl::event is only a placeholder.
//
// The consequence is that such an event is invisible to the device: it must never be used as a
// dependency of an enqueued command before it is set, otherwise that command would not be ordered
// against whatever the user event stands for. get() asserts on that to catch the misuse early.
//
// The queue stamp stays 0 because nothing is ever enqueued for this event, so
// sycl_stream::sync_events() skips it in barrier mode. That is correct: a barrier waits for work
// already submitted to the queue, and a user event has none. Only the paths which actually ask for
// the native ::sycl::event need the guard above.
struct sycl_user_event : public sycl_base_event {
public:
    explicit sycl_user_event(bool is_set = false) : sycl_base_event(0) {
        if (is_set) {
            set();
        }
    }

    void set_impl() override;
    bool get_profiling_info_impl(std::list<instrumentation::profiling_interval>& info) override;
    ::sycl::event& get() override {
        OPENVINO_ASSERT(_set,
                        "[GPU] Can't get native ::sycl::event from a user event which is not set yet. "
                        "SYCL runtime can't represent an incomplete event, thus a user event must be set "
                        "before anything asks for its native counterpart.");
        return _event;
    }

protected:
    instrumentation::timer<> _timer;
    std::unique_ptr<instrumentation::profiling_period_basic> _duration;

    // Always in the "complete" state. get() only hands it out once the event is set, so by then
    // that state matches the event it stands for.
    ::sycl::event _event;

private:
    void wait_impl() override;
    bool is_set_impl() override;
};

}  // namespace sycl
}  // namespace cldnn
