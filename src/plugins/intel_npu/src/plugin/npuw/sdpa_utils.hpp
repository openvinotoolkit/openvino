// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

// Shared SDPA (Scaled Dot-Product Attention) graph-analysis utilities.
//
// This header intentionally has NO dependency on pyramid_attention.hpp or
// host_flash_attention.hpp so that both can include it independently.
// The two attention strategies (Pyramid and HFA) both need to analyse the
// same decomposed-SDPA graph pattern, so the detection logic lives here.

#include <memory>
#include <vector>

#include "logging.hpp"
#include "openvino/openvino.hpp"

namespace ov {
namespace npuw {
namespace function {

// Structure to hold SDPA pattern nodes.
//
// After SplitKVCacheIntoBlocks transformation the single past_key / past_value
// parameter is replaced by N block parameters, so both fields are vectors.
// For the unmodified (non-block) case each vector will contain exactly one
// element (the original parameter node).
struct SDPAPatternNodes {
    std::shared_ptr<ov::Node> matmul1_node = nullptr;
    std::shared_ptr<ov::Node> matmul2_node = nullptr;
    std::shared_ptr<ov::Node> softmax_node = nullptr;
    std::shared_ptr<ov::Node> add_node = nullptr;
    std::vector<std::shared_ptr<ov::Node>> past_key_param_nodes;    // 1 (contiguous) or N (block-split) elements
    std::vector<std::shared_ptr<ov::Node>> past_value_param_nodes;  // 1 (contiguous) or N (block-split) elements
    std::shared_ptr<ov::Node> past_key_concat_node = nullptr;
    std::shared_ptr<ov::Node> past_value_concat_node = nullptr;

    bool is_valid() const {
        return matmul1_node && matmul2_node && softmax_node && add_node && !past_key_param_nodes.empty() &&
               !past_value_param_nodes.empty() && past_key_concat_node && past_value_concat_node;
    }

    // Log pattern information for debugging
    void log_pattern() const {
        LOG_DEBUG("SDPA Pattern nodes:");
        LOG_DEBUG("  MatMul1: " << (matmul1_node ? matmul1_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Add: " << (add_node ? add_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Softmax: " << (softmax_node ? softmax_node->get_friendly_name() : "null"));
        LOG_DEBUG("  MatMul2: " << (matmul2_node ? matmul2_node->get_friendly_name() : "null"));
        LOG_DEBUG("  Key Concat: " << (past_key_concat_node ? past_key_concat_node->get_friendly_name() : "null"));
        LOG_DEBUG(
            "  Value Concat: " << (past_value_concat_node ? past_value_concat_node->get_friendly_name() : "null"));
    }
};

// Find the decomposed SDPA sub-graph pattern (MatMul->Add->Softmax->MatMul) in
// a model and return all relevant nodes.  Returns an invalid result if the
// pattern is not found.
SDPAPatternNodes find_sdpa_pattern_nodes(const std::shared_ptr<ov::Model>& model);

// Traverse upward from an Add node's mask input to find the Parameter that
// supplies the attention mask.  Only unary ops are traversed; returns nullptr
// if the traversal fails.
std::shared_ptr<ov::op::v0::Parameter> find_mask_parameter(const std::shared_ptr<ov::Node>& add_node);

}  // namespace function
}  // namespace npuw
}  // namespace ov
