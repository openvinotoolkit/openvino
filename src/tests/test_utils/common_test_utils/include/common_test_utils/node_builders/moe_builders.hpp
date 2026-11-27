// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <utility>
#include <vector>

#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_output.hpp"
#include "openvino/core/partial_shape.hpp"
#include "openvino/core/shape.hpp"

namespace ov {
namespace test {

struct MoePatternParams {
    ov::PartialShape data_shape;
    size_t topk;
    size_t number_of_experts;
    size_t intermediate_size;
};

enum class MoERoutingType {
    SOFTMAX,       ///< Softmax -> TopK -> normalize routing
    SIGMOID_BIAS,  ///< Sigmoid -> Add(bias) -> TopK routing
};

enum class MoEActivationType {
    SWISH,     ///< Swish gate activation (SwiGLU)
    GELU,      ///< Gelu gate activation with Tanh approximation (GeGLU-Tanh)
    GELU_ERF,  ///< Gelu gate activation with ERF (exact) formula (GeGLU-ERF)
};

/// Softmax branch:
///   routing_weights -> Softmax -> TopK -> ReduceSum -> Divide (norm)
///   [-> Multiply(norm, Gather(per_expert_scale, topk_idx))  when use_per_expert_scale=true]
///   -> ScatterElementsUpdate -> Transpose -> Reshape -> Unsqueeze
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_softmax_routing_subgraph(
    const ov::Output<ov::Node>& routing_weights,
    size_t number_of_experts,
    size_t topk,
    bool use_per_expert_scale = false);

/// Sigmoid+bias branch:
///   routing_weights -> Sigmoid -> Add(bias) -> TopK -> Convert(i32)
///   -> GatherElements -> normalize -> ScatterElementsUpdate -> Transpose -> Reshape -> Unsqueeze
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_sigmoid_bias_routing_subgraph(
    const ov::Output<ov::Node>& routing_weights,
    ov::element::Type data_precision,
    size_t number_of_experts,
    size_t topk);

// gate_idx: 0 = gate (swish) at even positions (real gpt-oss), 1 = at odd positions.
std::shared_ptr<ov::Model> initMoE2GeMMSubgraph(
    const MoePatternParams& moe_params,
    const ov::element::Type data_precision,
    const ov::element::Type weights_precision,
    const bool use_weight_decompression = false,
    const std::optional<ov::element::Type> decompression_precision = std::nullopt,
    const std::optional<ov::element::Type> scale_precision = std::nullopt,
    const std::optional<ov::test::utils::DecompressionType> decompression_multiply_type = std::nullopt,
    const std::optional<ov::test::utils::DecompressionType> decompression_subtract_type = std::nullopt,
    const std::optional<bool> reshape_on_decompression = std::nullopt,
    const std::optional<int> decompression_group_size = std::nullopt,
    size_t gate_idx = 0);

std::shared_ptr<ov::Model> initMoE3GeMMSubgraph(
    const MoePatternParams& moe_params,
    const ov::element::Type data_precision,
    const ov::element::Type weights_precision,
    const bool use_weight_decompression = false,
    const std::optional<ov::element::Type> decompression_precision = std::nullopt,
    const std::optional<ov::element::Type> scale_precision = std::nullopt,
    const std::optional<ov::test::utils::DecompressionType> decompression_multiply_type = std::nullopt,
    const std::optional<ov::test::utils::DecompressionType> decompression_subtract_type = std::nullopt,
    const std::optional<bool> reshape_on_decompression = std::nullopt,
    const std::optional<int> decompression_group_size = std::nullopt,
    MoERoutingType routing_type = MoERoutingType::SOFTMAX,
    MoEActivationType activation_type = MoEActivationType::SWISH,
    bool use_per_expert_scale = false,
    bool use_layernorm_multiply = false);

struct MoEExpertSortOptions {
    bool token_index = true;          ///< build the token of each sorted row
    bool inverse_permutation = true;  ///< build the argsort of the permutation
    bool descending_argsort = false;  ///< the argsort TopK mode is MAX instead of MIN
    /// the scatter-add updates are Slice(Broadcast(1)), as exported by the PyTorch frontend
    bool slice_on_updates = false;
    bool permutation_to_i32 = false;  ///< the permutation is converted to i32 before its users
    /// top_k the token source is expanded to, the one of topk_ids by default
    std::optional<size_t> token_source_top_k;
    /// tensor whose dim 0 is the number of tokens of the token source, topk_ids by default
    ov::Output<ov::Node> token_source_tokens;
};

struct MoEExpertSortSubgraph {
    ov::Output<ov::Node> permutation;          ///< argsort of the flattened topk_ids
    ov::Output<ov::Node> offsets;              ///< inclusive end row of each expert (GroupedMatMul offsets)
    ov::Output<ov::Node> token_index;          ///< token of each sorted row, if requested
    ov::Output<ov::Node> inverse_permutation;  ///< argsort of the permutation, if requested
};

/// Expert sort of a grouped experts forward (transformers `grouped_mm_experts_forward`, optimum-intel `ov_grouped_mm`):
///   flat_ids = Reshape(topk_ids [T, K], [-1]), perm = TopK(flat_ids, k = T*K, min).indices
///   offsets  = CumSum(ScatterElementsUpdate(zeros[E], flat_ids[perm], ones, sum))
///   token    = Reshape(Broadcast(Unsqueeze(Range(0, T, 1)), [1, K]), [-1])[perm]
///   inv_perm = TopK(perm, k = T*K, min).indices
MoEExpertSortSubgraph build_moe_expert_sort_subgraph(const ov::Output<ov::Node>& topk_ids,
                                                     size_t number_of_experts,
                                                     const MoEExpertSortOptions& options = {});

}  // namespace test
}  // namespace ov
