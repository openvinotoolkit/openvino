// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "cldnn/runtime/event.hpp"
#include "cldnn/runtime/stream.hpp"
#include "sycl_common.hpp"
#include "sycl_engine.hpp"

#include <memory>
#include <chrono>
#include <thread>
#include <iostream>
#include <sstream>
#include <utility>
#include <vector>

namespace cldnn {
namespace sycl {

class sycl_stream : public stream {
public:
    cl::sycl::queue& queue() { return _command_queue; }

    explicit sycl_stream(const sycl_engine& engine);
    sycl_stream(sycl_stream&& other)
        : stream(other._engine.configuration().queue_type),
          _engine(other._engine),
          _command_queue(other._command_queue),
          _queue_counter(other._queue_counter.load()),
          _last_barrier(other._last_barrier.load()),
          _last_barrier_ev(other._last_barrier_ev),
          _output_event(other._output_event) {}

    ~sycl_stream() = default;

    void sync_events(std::vector<event::ptr> const& deps, bool is_output_event = false) override;
    void release_pending_memory();
    void flush() const override;
    void finish() const override;

    void set_output_event(bool out_event) { _output_event = out_event; }

    void set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) override;
    event::ptr enqueue_kernel(kernel& kernel,
                              const kernel_arguments_desc& args_desc,
                              const kernel_arguments_data& args,
                              std::vector<event::ptr> const& deps,
                              bool is_output_event = false) override;
    event::ptr enqueue_marker(std::vector<event::ptr> const& deps, bool is_output_event) override;
    event::ptr group_events(std::vector<event::ptr> const& deps) override;
    void wait_for_events(const std::vector<event::ptr>& events) override;
    void enqueue_barrier() override;
    void reset_events() override;
    event::ptr create_user_event(bool set) override;
    event::ptr create_base_event() override;
    void release_events_pool() override;
    const sycl_engine& get_sycl_engine() const { return _engine; };

private:
    const sycl_engine& _engine;
    // queue is mutable as finish() method in sycl is not marked as const for some reason
    mutable cl::sycl::queue _command_queue;
    std::atomic<uint64_t> _queue_counter{0};
    std::atomic<uint64_t> _last_barrier{0};
    cl::sycl::event _last_barrier_ev;
    bool _output_event = false;
};

}  // namespace sycl
}  // namespace cldnn
