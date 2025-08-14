// Copyright (C) 2019-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
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
    ::sycl::queue& get_sycl_queue() { return _command_queue; }

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
    void finish() override;
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

    // const sycl::UsmHelper& get_usm_helper() const { return _engine.get_usm_helper(); }

    static QueueTypes detect_queue_type(void* queue_handle);

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream() override;
#endif

private:
    void sync_events(std::vector<event::ptr> const& deps, bool is_output = false);

    const sycl_engine& _engine;
    ::sycl::queue _command_queue;
    std::atomic<uint64_t> _queue_counter{0};
    std::atomic<uint64_t> _last_barrier{0};
    ::sycl::event _last_barrier_ev;

#ifdef ENABLE_ONEDNN_FOR_GPU
    std::shared_ptr<dnnl::stream> _onednn_stream = nullptr;
#endif
};

}  // namespace sycl
}  // namespace cldnn
