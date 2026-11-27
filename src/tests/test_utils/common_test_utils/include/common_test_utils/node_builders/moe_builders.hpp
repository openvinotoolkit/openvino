// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <memory>
#include <optional>
#include <tuple>
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

/// Expert sort of a grouped experts forward (transformers `grouped_mm_experts_forward`, optimum-intel `ov_grouped_mm`).
/// @param topk_ids                  Router expert ids [T, K].
/// @param number_of_experts         Number of experts E.
/// @param with_token                Build the token index of each sorted row.
/// @param with_inverse_permutation  Build the argsort of the permutation.
/// @param descending_argsort        Argsort with TopK mode MAX instead of MIN.
/// @param slice_on_updates          Scatter-add updates are Slice(Broadcast(1)), as exported by the PyTorch frontend.
/// @param permutation_to_i32        Convert the permutation to i32 before its users.
/// @param token_source_top_k        top_k the token source is expanded to, the one of topk_ids if not set.
/// @param token_source_tokens       Tensor whose dim 0 is the number of tokens of the token source, topk_ids if empty.
/// @return {perm, offsets, token, inv_perm}:
///         perm     - argsort of the flattened topk_ids [T*K],
///         offsets  - inclusive end row of each expert [E] (GroupedMatMul offsets),
///         token    - token of each sorted row [T*K], empty if !with_token,
///         inv_perm - argsort of perm [T*K], empty if !with_inverse_permutation.
std::tuple<ov::Output<ov::Node>, ov::Output<ov::Node>, ov::Output<ov::Node>, ov::Output<ov::Node>>
build_moe_expert_sort_subgraph(const ov::Output<ov::Node>& topk_ids,
                               size_t number_of_experts,
                               bool with_token = true,
                               bool with_inverse_permutation = true,
                               bool descending_argsort = false,
                               bool slice_on_updates = false,
                               bool permutation_to_i32 = false,
                               const std::optional<size_t>& token_source_top_k = std::nullopt,
                               const ov::Output<ov::Node>& token_source_tokens = {});

}  // namespace test
}  // namespace ov
