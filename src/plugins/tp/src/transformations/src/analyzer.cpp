// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <unordered_set>

#include "block_utils.hpp"
#include "openvino/core/except.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/pass/manager.hpp"
#include "tp_transformations/analyzer.hpp"
#include "tp_transformations/verifier.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace tp {

namespace {

size_t weight_bytes(const std::shared_ptr<ov::Node>& matmul) {
    size_t bytes = 0;
    std::unordered_set<ov::Node*> visited;
    ov::op::util::visit_path(
        matmul->get_input_node_ptr(1),
        visited,
        [&](ov::Node* node) {
            if (const auto* c = ov::as_type<ov::op::v0::Constant>(node))
                bytes += c->get_byte_size();
        },
        [](ov::Node*) {
            return false;
        });
    return bytes;
}

}  // namespace

TPLayerAnalyzer::TPLayerAnalyzer(const TPAnalyzerConfig& config, TPAnalysisReportPtr report)
    : m_config(config),
      m_report(report ? std::move(report) : std::make_shared<TPAnalysisReport>()) {}

bool TPLayerAnalyzer::run_on_model(const std::shared_ptr<ov::Model>& model) {
    OPENVINO_ASSERT(m_config.world_size >= 2, "[TP] the analysis needs at least 2 ranks");

    ov::pass::Manager manager("TPLayerAnalyzer");
    manager.set_per_pass_validation(false);
    manager.register_pass<TPPagedAttentionBlock>(m_config, m_report);
    manager.register_pass<TPStatefulSDPABlock>(m_config, m_report);
    manager.register_pass<TPGatedMLPBlock>(m_config, m_report);
    manager.register_pass<TPDenseMLPBlock>(m_config, m_report);
    if (m_config.shard_lm_head)
        manager.register_pass<TPVocabParallelBlock>(m_config, m_report);
    manager.run_passes(model);

    OPENVINO_ASSERT(m_report->attention_blocks > 0,
                    "[TP] Could not identify transformer layers for TP: the model contains neither "
                    "PagedAttentionExtension nor ScaledDotProductAttention with recognizable projections. "
                    "Model may not be supported.");
    OPENVINO_ASSERT(m_report->mlp_blocks > 0, "[TP] Could not locate an MLP block. Model may not be supported.");

    for (const auto& node : model->get_ordered_ops()) {
        if (blocks::is_linear_layer(node.get()) && get_role(*node) == nullptr) {
            m_report->unclaimed.push_back(node->get_friendly_name());
            m_report->unclaimed_bytes += weight_bytes(node);
        }
    }
    TPShardingVerifier(m_config.world_size).run_on_model(model);
    return true;
}

}  // namespace tp
}  // namespace ov
