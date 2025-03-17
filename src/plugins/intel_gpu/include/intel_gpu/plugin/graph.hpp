// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#ifndef NOMINMAX
# define NOMINMAX
#endif

#include "intel_gpu/graph/network.hpp"
#include "intel_gpu/plugin/remote_context.hpp"
#include "intel_gpu/plugin/program_builder.hpp"

#include <vector>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <vector>
#include <condition_variable>

namespace ov::intel_gpu {
struct HostTimeProfilingEntry {
    int64_t inputs_processing = 0;
    int64_t enqueue = 0;
    int64_t wait = 0;
    int64_t outputs_processing = 0;
};

class Graph final {
public:
    using Ptr = std::shared_ptr<Graph>;
    enum class Stage : uint32_t {
        PREPROC = 1,
        EXECUTE = 2,
        POSTPROC = 4
    };

    Graph(std::shared_ptr<ov::Model> model, const RemoteContextImpl::Ptr& context, const ExecutionConfig& config, uint16_t stream_id = 0);
    Graph(cldnn::BinaryInputBuffer& ib, const RemoteContextImpl::Ptr& context, const ExecutionConfig& config, uint16_t stream_id = 0);
    Graph(std::shared_ptr<Graph> graph, uint16_t stream_id = 0);
    ~Graph();

    void export_model(cldnn::BinaryOutputBuffer &ob);
    std::shared_ptr<ov::Model> get_runtime_model();

    bool is_loaded() const;

    std::vector<ov::ProfilingInfo> get_profiling_info() const;
    void update_profiling_info();

    cldnn::engine& get_engine() const { return m_context->get_engine(); }
    const ExecutionConfig& get_config() const { return m_config; }

    const std::map<size_t, cldnn::layout>& get_input_layouts() const { return m_input_layouts; }
    std::shared_ptr<cldnn::network> get_network() const;

    std::vector<cldnn::primitive_id> input_port_index_to_internal(size_t input_port_index) const;
    std::string out_port_index_to_internal(size_t out_port_index) const;

    void wait(Stage stage_mask) {
        std::unique_lock<std::mutex> lock(m_infer_mutex);
        m_cv.wait(lock, [&stage_mask, this] {
            return (m_state & (uint32_t)stage_mask) == 0;
        });
        m_state |= (uint32_t)stage_mask;
    }

    void notify(Stage stage_mask) {
        {
            std::lock_guard<std::mutex> lock(m_infer_mutex);
            m_state &= ~(uint32_t)stage_mask;
        }
        m_cv.notify_one();
    }
    std::mutex& get_mutex() { return m_infer_mutex; }

    bool use_external_queue() const;

    std::vector<HostTimeProfilingEntry> host_exec_times;

private:
    RemoteContextImpl::Ptr m_context;
    ExecutionConfig m_config;
    uint16_t m_stream_id;
    uint32_t m_state = 0;
    std::condition_variable m_cv;
    std::mutex m_infer_mutex;

    std::shared_ptr<cldnn::network> m_network;
    std::map<std::string, cldnn::primitive_id> primitiveIDs;
    std::map<size_t, std::vector<cldnn::primitive_id>> inputPrimitiveIDs;
    std::map<size_t, cldnn::primitive_id> prevPrimitiveIDs;

    std::map<cldnn::primitive_id, std::pair<std::string, PerfCounter>> perfMap;
    std::vector<cldnn::primitive_id> profilingIDs;

    std::map<size_t, cldnn::layout> m_input_layouts;

    void build(std::shared_ptr<cldnn::program> program);
    std::shared_ptr<ov::Model> get_runtime_model(std::vector<cldnn::primitive_info>& pi, bool filter_const_primitives = true);
};

}  // namespace ov::intel_gpu
