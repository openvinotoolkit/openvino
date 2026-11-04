// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <cstddef>
#include <memory>
#include <string>
#include <vector>

#include "openvino/pass/matcher_pass.hpp"
#include "openvino/pass/pass.hpp"

namespace ov {
namespace tp {

struct TPAnalyzerConfig {
    size_t world_size = 2;
    /// Split the vocabulary projection and gather it into rank 0.
    bool shard_lm_head = true;
};

/// What the analysis found, for the plugin's log.
struct TPAnalysisReport {
    size_t attention_blocks = 0;
    size_t mlp_blocks = 0;
    bool lm_head = false;
    /// Weights cut by a Slice at run time: their chain cannot be cut at the rank boundaries.
    std::vector<std::string> runtime_slices;
    /// Linear layers no block pass claimed; they stay whole on every rank.
    std::vector<std::string> unclaimed;
    size_t unclaimed_bytes = 0;
    /// Geometry of the first attention block.
    int64_t num_heads = 0;
    int64_t num_kv_heads = 0;
    int64_t head_dim = 0;
    int64_t intermediate_size = 0;
    /// Variables split along the KV-head axis.
    std::vector<std::string> sharded_variables;
    /// Consecutive ranks sharing one KV head: world_size / num_kv_heads with fewer KV heads than ranks.
    size_t kv_replicas = 1;
};

using TPAnalysisReportPtr = std::shared_ptr<TPAnalysisReport>;

/// Pass 1: the block passes, then the sharding verifier. Leaves tp_shard / tp_role annotations.
class TPLayerAnalyzer : public ov::pass::ModelPass {
public:
    OPENVINO_MODEL_PASS_RTTI("TPLayerAnalyzer");
    explicit TPLayerAnalyzer(const TPAnalyzerConfig& config, TPAnalysisReportPtr report = nullptr);
    bool run_on_model(const std::shared_ptr<ov::Model>& model) override;

private:
    TPAnalyzerConfig m_config;
    TPAnalysisReportPtr m_report;
};

/// PagedAttentionExtension with separate Q/K/V projections and the O projection.
class TPPagedAttentionBlock : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TPPagedAttentionBlock");
    TPPagedAttentionBlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report);
};

/// ScaledDotProductAttention with separate Q/K/V and O projections; KV cache in variables.
class TPStatefulSDPABlock : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TPStatefulSDPABlock");
    TPStatefulSDPABlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report);
};

/// down(act(gate(x)) * up(x)).
class TPGatedMLPBlock : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TPGatedMLPBlock");
    TPGatedMLPBlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report);
};

/// down(act(up(x))).
class TPDenseMLPBlock : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TPDenseMLPBlock");
    TPDenseMLPBlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report);
};

/// The vocabulary projection in front of a Result, split by vocabulary and gathered into rank 0.
class TPVocabParallelBlock : public ov::pass::MatcherPass {
public:
    OPENVINO_MATCHER_PASS_RTTI("TPVocabParallelBlock");
    TPVocabParallelBlock(const TPAnalyzerConfig& config, TPAnalysisReportPtr report);
};

}  // namespace tp
}  // namespace ov
