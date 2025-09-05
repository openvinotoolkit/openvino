// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include "intel_gpu/runtime/event.hpp"
#include "intel_gpu/runtime/stream.hpp"
#include "ze_common.hpp"
#include "ze_engine.hpp"
#include "ze_event.hpp"

namespace cldnn {
namespace ze {

class ze_stream : public stream {
public:
    ze_command_list_handle_t get_queue() const { return m_command_list; }

    ze_stream(const ze_engine& engine, const ExecutionConfig& config);
    ze_stream(ze_stream&& other)
        : stream(other.m_queue_type, other.m_sync_method)
        , _engine(other._engine)
        , m_command_list(other.m_command_list)
        , m_queue_counter(other.m_queue_counter.load())
        , m_last_barrier(other.m_last_barrier.load())
        , m_last_barrier_ev(other.m_last_barrier_ev)
        , m_pool(other.m_pool) {}

    ~ze_stream();

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

#ifdef ENABLE_ONEDNN_FOR_GPU
    dnnl::stream& get_onednn_stream() override;
#endif

private:
    void sync_events(std::vector<event::ptr> const& deps, bool is_output = false);

    const ze_engine& _engine;
    mutable ze_command_list_handle_t m_command_list = 0;
    mutable std::atomic<uint64_t> m_queue_counter{0};
    std::atomic<uint64_t> m_last_barrier{0};
    std::shared_ptr<ze_event> m_last_barrier_ev = nullptr;
    ze_events_pool m_pool;

#ifdef ENABLE_ONEDNN_FOR_GPU
    std::shared_ptr<dnnl::stream> _onednn_stream = nullptr;
#endif
};

}  // namespace ze
}  // namespace cldnn
