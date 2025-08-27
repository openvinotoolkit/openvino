
// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "executable.hpp"
#include "openvino/core/node.hpp"
#include "openvino/itt.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"

// NEW: cache glue
#include "cache/cache_eviction.hpp"
#include "cache/cache_manager.hpp"

namespace ov {
namespace template_plugin {

// ---- helper structs (namespace scope, not inside class) ----
struct ScoresPort {
    size_t result_index{};  // which Result consumes PagedAttention::output(1)
    size_t layer_id{};      // logical decoder layer id (0..L-1)
};

struct LayerPorts {
    ov::Output<const ov::Node> k_param;  // PagedAttention input(3)
    ov::Output<const ov::Node> v_param;  // PagedAttention input(4)
};

// forward declaration
class CompiledModel;

// ! [infer_request:header]
class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::template_plugin::CompiledModel>& compiled_model);
    ~InferRequest();

    void infer() override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    // pipeline methods-stages which are used in async infer request implementation and assigned to particular executor
    void infer_preprocess();
    void start_pipeline();
    void wait_pipeline();
    void infer_postprocess();
    void cancel();

    void set_tensors_impl(const ov::Output<const ov::Node> port,
                          const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

private:
    std::shared_ptr<const CompiledModel> get_template_model() const;

    // helper glue for KV binding and scores harvesting
    void ensure_kv_cache_bound();
    void register_scores_and_evict();

    enum { Preprocess, Postprocess, StartPipeline, WaitPipeline, numOfStages };

    std::array<openvino::itt::handle_t, numOfStages> m_profiling_task;
    // for performance counters
    std::array<std::chrono::duration<float, std::micro>, numOfStages> m_durations;

    std::vector<ov::Tensor> m_backend_input_tensors;
    std::vector<ov::Tensor> m_backend_output_tensors;
    std::shared_ptr<ov::runtime::Executable> m_executable;
    ov::EvaluationContext m_eval_context;
    std::vector<ov::SoPtr<ov::IVariableState>> m_variable_states;

    // ---- NEW: cache management + eviction ----
    std::shared_ptr<ov::cache::CacheManager> m_cache_mgr;
    std::unique_ptr<ov::cache::CacheEvictionAlgorithm> m_eviction;
    size_t m_cfg_max_kv_blocks{0};  // optional knob; 0 == allocate on demand

    std::vector<ScoresPort> m_scores_ports;  // Result indices consuming scores (output 1)
    std::vector<LayerPorts> m_layer_ports;   // KC/VC binding ports (inputs 3,4 of PagedAttention)
};
// ! [infer_request:header]

}  // namespace template_plugin
}  // namespace ov
