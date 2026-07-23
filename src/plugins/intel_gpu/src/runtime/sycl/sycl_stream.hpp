// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Copyright (C) 2026 FUJITSU LIMITED
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "sycl_common.hpp"
#include "sycl_engine.hpp"

#include <memory>
#include <vector>

namespace cldnn {
namespace sycl {

class sycl_stream : public stream {
public:
    ::sycl::queue& get_sycl_queue() const { return _command_queue; }

    sycl_stream(const sycl_engine& engine, const ExecutionConfig& config);
    sycl_stream(const sycl_engine& engine, const ExecutionConfig& config, void* handle);
    sycl_stream(sycl_stream&& other)
        : stream(other.m_queue_type, other.m_sync_method)
        , _engine(other._engine)
        , _command_queue(other._command_queue)
        , _queue_counter(other._queue_counter.load())
        , _last_barrier(other._last_barrier.load())
        , _last_barrier_ev(other._last_barrier_ev) {}

    ~sycl_stream() = default;

    void flush() const override;
    void finish() const override;
    void wait() override;

    void set_arguments(kernel& kernel, const kernel_arguments_desc& args_desc, const kernel_arguments_data& args) override;
    event::ptr enqueue_kernel(kernel& kernel,
                              const kernel_arguments_desc& args_desc,
                              const kernel_arguments_data& args,
                              std::vector<event::ptr> const& deps,
                              bool is_output = false) override;
    event::ptr enqueue_marker(std::vector<event::ptr> const& deps, bool is_output) override;
    event::ptr group_events(std::vector<event::ptr> const& deps) override;
    void wait_for_events(const std::vector<event::ptr>& events) override;
    void enqueue_barrier() override;
    event::ptr create_user_event(bool set) override;
    event::ptr create_base_event() override;
    event::ptr create_base_event(::sycl::event& event);
    std::unique_ptr<surfaces_lock> create_surfaces_lock(const std::vector<memory::ptr> &mem) const override;

    static QueueTypes detect_queue_type(void* queue_handle);

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream() override;
#endif

private:
    void sync_events(std::vector<event::ptr> const& deps, bool is_output = false);

    const sycl_engine& _engine;
    mutable ::sycl::queue _command_queue;
    std::atomic<uint64_t> _queue_counter{0};
    std::atomic<uint64_t> _last_barrier{0};
    ::sycl::event _last_barrier_ev;

#ifdef ENABLE_ONEDNN_FOR_GPU
    std::shared_ptr<dnnl::stream> _onednn_stream = nullptr;
#endif
};

}  // namespace sycl
}  // namespace cldnn
