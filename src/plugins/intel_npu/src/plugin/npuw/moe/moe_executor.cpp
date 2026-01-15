// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "moe_executor.hpp"

#include "../compiled_model.hpp"  // For CompiledModel::CompiledModelDesc
#include "../logging.hpp"
#include "../moe_infer_utils.hpp"
#include "openvino/core/except.hpp"

namespace ov {
namespace npuw {
namespace moe {

MoEExecutor::MoEExecutor(ISubrequestAccessor& accessor, ProfilerFn profiler, AllocatorFn allocator)
    : m_accessor(accessor),
      m_profiler(std::move(profiler)),
      m_allocator(std::move(allocator)) {
    LOG_DEBUG("MoEExecutor created");
}

void MoEExecutor::prepare(size_t real_idx, const void* compiled_model_desc_ptr, size_t pool_size) {
    LOG_INFO("Preparing MoE resources for submodel[" << real_idx << "]...");
    LOG_BLOCK();

    // Cast to actual type (avoiding circular dependency in header)
    const auto& desc = *static_cast<const ov::npuw::CompiledModel::CompiledModelDesc*>(compiled_model_desc_ptr);

    if (!desc.moe_experts.has_value()) {
        OPENVINO_THROW("MoEExecutor::prepare called on non-MoE submodel[", real_idx, "]");
    }

    const auto& moe_experts = desc.moe_experts.value();

    // Step 1: Build MoEConfig from CompiledModelDesc
    MoEConfig config;
    config.num_experts = moe_experts.num_experts;
    config.num_active_experts = moe_experts.num_active_experts;
    config.input_token_count = moe_experts.input_token_count;
    config.expert_hidden_dim = moe_experts.expert_hidden_dim;
    config.param_mapping = moe_experts._param_mapping;
    config.router_scores_idx = moe_experts._router_scores_idx;
    config.expert_input_param_idx = moe_experts._expert_input_param_idx;
    config.compiled_models = moe_experts._compiled_models;

    // Validate configuration
    if (!config.is_valid()) {
        OPENVINO_THROW("Invalid MoE configuration for submodel[", real_idx, "]");
    }

    LOG_DEBUG("MoE Config: num_experts=" << config.num_experts << ", num_active_experts=" << config.num_active_experts
                                         << ", input_token_count=" << config.input_token_count
                                         << ", mode=" << (config.is_decoding() ? "DECODING" : "PREFILL"));

    m_configs[real_idx] = std::move(config);

    // Step 2: Initialize MoEResources
    MoEResources resources;
    resources.initialize(m_configs[real_idx], m_allocator, pool_size, get_device_name(real_idx));

    m_resources[real_idx] = std::move(resources);

    LOG_INFO("MoE preparation completed for submodel[" << real_idx << "]");
}

void MoEExecutor::run(size_t real_idx, size_t idx, const MoEIO& io) {
    LOG_DEBUG("MoEExecutor::run() - submodel[" << real_idx << "], idx[" << idx << "]");

    // TODO: Phase 2 - Implement actual execution logic
    // Will be migrated from run_moe_infer() in Phase 2
    OPENVINO_THROW("MoEExecutor::run() not yet implemented - Phase 2 task");
}

void MoEExecutor::recreate_requests(size_t real_idx) {
    LOG_INFO("Recreating MoE requests for submodel[" << real_idx << "]...");

    // Reset resources (will be re-populated on next inference)
    if (m_resources.find(real_idx) != m_resources.end()) {
        m_resources[real_idx].reset();
    }

    LOG_INFO("MoE requests recreated for submodel[" << real_idx << "]");
}

void MoEExecutor::run_batch_experts(size_t idx,
                                    size_t real_idx,
                                    const std::vector<size_t>& selected_experts,
                                    const MoEIO& io) {
    // TODO: Phase 2 - Migrate from run_moe_batch_experts_inference()
    OPENVINO_THROW("run_batch_experts() not yet implemented - Phase 2 task");
}

void MoEExecutor::run_iterative_experts(size_t idx,
                                        size_t real_idx,
                                        const std::vector<size_t>& selected_experts,
                                        const MoEIO& io) {
    // TODO: Phase 2 - Migrate from run_moe_iterative_experts_inference()
    OPENVINO_THROW("run_iterative_experts() not yet implemented - Phase 2 task");
}

void MoEExecutor::set_router_scores(size_t idx,
                                    size_t real_idx,
                                    const std::vector<size_t>& selected_experts,
                                    RqPtr& request,
                                    const MoEIO& io) {
    // TODO: Phase 2 - Migrate from set_unrolled_router_scores()
    OPENVINO_THROW("set_router_scores() not yet implemented - Phase 2 task");
}

std::string MoEExecutor::get_device_name(size_t real_idx) const {
    const auto& desc = *static_cast<const CompiledModel::CompiledModelDesc*>(m_accessor.get_submodel_desc(real_idx));
    return *desc.device_it;
}

}  // namespace moe
}  // namespace npuw
}  // namespace ov
