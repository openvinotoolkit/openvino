// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#pragma once

#include <array>
#include <chrono>
#include <map>
#include <memory>
#include <mutex>
#include <set>
#include <string>
#include <vector>

#include "cache_eviction.hpp"
#include "cache_manager.hpp"
#include "executable.hpp"
#include "openvino/core/node.hpp"
#include "openvino/itt.hpp"
#include "openvino/runtime/isync_infer_request.hpp"
#include "openvino/runtime/ivariable_state.hpp"

namespace ov {
namespace template_plugin {

class CompiledModel;

struct ScoresPort {
    size_t result_index;
    size_t layer_id;
};
class ScoresLocator {
public:
    static std::vector<ScoresPort> find(const std::shared_ptr<const CompiledModel>& cm);
};

// ! [infer_request:header]
class InferRequest : public ov::ISyncInferRequest {
public:
    explicit InferRequest(const std::shared_ptr<const ov::template_plugin::CompiledModel>& compiled_model);
    ~InferRequest();

    void infer() override;
    std::vector<ov::SoPtr<ov::IVariableState>> query_state() const override;
    std::vector<ov::ProfilingInfo> get_profiling_info() const override;

    // pipeline methods-stages
    void infer_preprocess();
    void start_pipeline();
    void wait_pipeline();
    void infer_postprocess();
    void cancel();

    void set_tensors_impl(const ov::Output<const ov::Node> port,
                          const std::vector<ov::SoPtr<ov::ITensor>>& tensors) override;

private:
    std::shared_ptr<const CompiledModel> get_template_model() const;

    enum { Preprocess, Postprocess, StartPipeline, WaitPipeline, numOfStages };

    std::array<openvino::itt::handle_t, numOfStages> m_profiling_task;
    std::array<std::chrono::duration<float, std::micro>, numOfStages> m_durations;

    std::vector<ov::Tensor> m_backend_input_tensors;
    std::vector<ov::Tensor> m_backend_output_tensors;
    std::shared_ptr<ov::runtime::Executable> m_executable;
    ov::EvaluationContext m_eval_context;
    std::vector<ov::SoPtr<ov::IVariableState>> m_variable_states;

    // ---- NEW ----
    std::shared_ptr<ov::cache::CacheManager> m_cache_mgr;
    std::unique_ptr<ov::cache::CacheEvictionAlgorithm> m_eviction;
    std::vector<ScoresPort> m_scores_ports;

    void ensure_kv_cache_bound();
    void register_scores_and_evict();
};

}  // namespace template_plugin
}  // namespace ov
