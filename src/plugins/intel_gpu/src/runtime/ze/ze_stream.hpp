// // Copyright (C) 2019-2021 Intel Corporation
// // SPDX-License-Identifier: Apache-2.0
// //

#pragma once

#include "cldnn/runtime/event.hpp"
#include "cldnn/runtime/stream.hpp"
#include "ze_common.hpp"
#include "ze_engine.hpp"
#include "ze_event.hpp"

#include <memory>
#include <chrono>
#include <thread>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace cldnn {
namespace ze {

// Possible sync methods for kernels in stream
enum class sync_methods {
    /* Build dependency graph using events. Each kernel creates proper cl_event which is set as dependency of users
       At this moment it requires multiple retain/release calls for cl_event after each enqueueNDRange
       which is less performant comparing to the barriers version
    */
    events = 0,
    /* Enqueue barriers between dependent kernels. For example consider the following dimond dependency graph:
             kernel_0
             /      \
         kernel_1   kernel_2
             \      /
             kernel_3
       In that case we do the following:
       1. Enqueue kernel_0
       2. Enqueue barrier (ensures kernel_0 is completed)
       3. Enqueue kernel_1
       4. Enqueue kernel_2 (doesn't depend on kernel_1)
       5. Enqueue barrier (ensures kernel_1 and kernel_2 are completed)
       6. Enqueue kernel_3
    */
    barriers = 1,
    /* No explicit syncronization is needed. Applicable for in-order queue only */
    none = 2
};

class ze_stream : public stream {
public:
    ze_command_list_handle_t get_queue() const { return _command_list; }

    explicit ze_stream(const ze_engine& engine);
    ze_stream(ze_stream&& other)
        : stream(other._engine.configuration().queue_type)
        , _engine(other._engine)
        , _command_list(other._command_list)
        , _queue_counter(other._queue_counter.load())
        , _last_barrier(other._last_barrier.load())
        , _last_barrier_ev(other._last_barrier_ev)
        , sync_method(other.sync_method) {}

    ~ze_stream() = default;

    void flush() const override;
    void finish() const override;

    void set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) override;
    event::ptr enqueue_kernel(kernel& kernel,
                              const kernel_arguments_desc& args_desc,
                              const kernel_arguments_data& args,
                              std::vector<event::ptr> const& deps,
                              bool is_output = false) override;
    ze_event::ptr enqueue_marker(std::vector<event::ptr> const& deps, bool is_output) override;
    event::ptr group_events(std::vector<event::ptr> const& deps) override;
    void wait_for_events(const std::vector<event::ptr>& events) override;
    void enqueue_barrier() override;
    ze_event::ptr create_user_event(bool set) override;
    event::ptr create_base_event() override;

private:
    void sync_events(std::vector<event::ptr> const& deps, bool is_output = false);

    const ze_engine& _engine;
    ze_command_list_handle_t  _command_list;
    ze_event_pool_handle_t _event_pool;
    std::atomic<uint64_t> _queue_counter{0};
    std::atomic<uint64_t> _last_barrier{0};
    ze_event_handle_t _last_barrier_ev;
    uint32_t event_idx = 0;
    sync_methods sync_method;
};

}  // namespace ze
}  // namespace cldnn
