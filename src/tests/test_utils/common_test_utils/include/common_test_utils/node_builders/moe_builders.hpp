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
    SOFTMAX,                ///< Softmax -> TopK -> normalize routing
    SIGMOID_BIAS,           ///< Sigmoid -> Add(bias) -> TopK routing
    SIGMOID_BIAS_SCALED_NORM, ///< Sigmoid -> Add(bias) -> TopK -> normalize -> Multiply(scale) routing
};

/// Softmax branch:
///   routing_weights -> Softmax -> TopK -> ReduceSum -> Divide (norm)
///   -> ScatterElementsUpdate -> Transpose -> Reshape -> Unsqueeze
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>
build_softmax_routing_subgraph(const ov::Output<ov::Node>& routing_weights, size_t number_of_experts, size_t topk);

/// Sigmoid+bias branch:
///   routing_weights -> Sigmoid -> Add(bias) -> TopK -> Convert(i32)
///   -> GatherElements -> normalize -> ScatterElementsUpdate -> Transpose -> Reshape -> Unsqueeze
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_sigmoid_bias_routing_subgraph(
    const ov::Output<ov::Node>& routing_weights,
    ov::element::Type data_precision,
    size_t number_of_experts,
    size_t topk);

/// Sigmoid+bias+scaled_norm branch — same as SIGMOID_BIAS but inserts
///   Multiply(Divide(...), Constant) between the normalization Divide and the Slice.
std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_sigmoid_bias_scaled_norm_routing_subgraph(
    const ov::Output<ov::Node>& routing_weights,
    ov::element::Type data_precision,
    size_t number_of_experts,
    size_t topk);

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
    const std::optional<int> decompression_group_size = std::nullopt);

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
    MoERoutingType routing_type = MoERoutingType::SOFTMAX);

}  // namespace test
}  // namespace ov
