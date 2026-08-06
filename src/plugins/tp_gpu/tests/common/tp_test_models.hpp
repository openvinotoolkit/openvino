// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Synthetic transformer blocks shared by the unit and functional suites, so
// that what the rewriter is checked against structurally is the same thing the
// accuracy tests run on hardware.

#pragma once

#include <memory>
#include <string>
#include <vector>

#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/op/ops.hpp"

namespace ov::tp_gpu::tests {

/// Geometry of the synthetic transformer block.
///
/// Deliberately awkward: `intermediate` is not a multiple of the world sizes
/// under test, so the split has to fall back on whole quantization groups, and
/// `num_kv_heads` is small enough that the grouped-query pairing is exercised
/// while still allowing every world size the tests use.
struct BlockConfig {
    size_t hidden = 256;
    size_t num_heads = 8;
    size_t num_kv_heads = 4;
    size_t head_dim = 32;
    size_t intermediate = 448;
    int group_size = 32;  ///< quantization group; -1 disables grouping
    ov::element::Type weights_precision = ov::element::u4;
    bool with_bias = false;
    /// Weight laid out as [out, in] and consumed with `transpose_b`, the way
    /// PyTorch exports it.  Set to false to exercise the [in, out] layout.
    bool transpose_b = true;
    /// down(act(gate(x)) * up(x)) when true, down(act(up(x))) otherwise.
    bool gated_mlp = true;
    /// Wrap K and V in a ReadValue/Assign KV cache.
    bool stateful = false;
};

/// A linear layer with a decompressed weight, matching how PyTorch exports one:
/// `MatMul(x, W, transpose_b=true)` over a `Constant -> Convert -> Subtract ->
/// Multiply -> Reshape` chain, with per-group scales and zero-points.
inline std::shared_ptr<ov::Node> make_projection(const ov::Output<ov::Node>& input,
                                                 size_t in_features,
                                                 size_t out_features,
                                                 const BlockConfig& config,
                                                 bool with_bias,
                                                 size_t seed) {
    auto weights = ov::test::utils::initMatMulDecompressionSubgraph(
        ov::Shape{in_features, out_features},
        config.group_size,
        ov::element::f32,                          // data precision
        config.weights_precision,                  // stored weight precision
        ov::element::f32,                          // decompression precision
        ov::element::f32,                          // scale precision
        config.transpose_b,                        // weight layout
        ov::test::utils::DecompressionType::full,  // per-group scales
        ov::test::utils::DecompressionType::full,  // per-group zero-points
        /*reshape_on_decompression_constant=*/true,
        /*insert_transpose_node=*/false,  // transpose_b on the MatMul instead
        seed);

    std::shared_ptr<ov::Node> projection =
        std::make_shared<ov::op::v0::MatMul>(input, weights, /*transpose_a=*/false, config.transpose_b);

    if (with_bias) {
        auto bias = ov::op::v0::Constant::create(ov::element::f32,
                                                 ov::Shape{1, 1, out_features},
                                                 std::vector<float>(out_features, 0.1f));
        projection = std::make_shared<ov::op::v1::Add>(projection, bias);
    }
    return projection;
}

inline std::shared_ptr<ov::Node> reshape_to_heads(const ov::Output<ov::Node>& input, size_t heads, size_t head_dim) {
    auto shape = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{0, 0, static_cast<int64_t>(heads), static_cast<int64_t>(head_dim)});
    return std::make_shared<ov::op::v1::Reshape>(input, shape, /*special_zero=*/true);
}

inline std::shared_ptr<ov::Node> transpose_heads(const ov::Output<ov::Node>& input) {
    auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{4}, std::vector<int64_t>{0, 2, 1, 3});
    return std::make_shared<ov::op::v1::Transpose>(input, order);
}

/// Expands KV heads up to the query head count the way exported models do:
/// Unsqueeze -> multiply by a tensor of ones -> Reshape.  The ones constant
/// broadcasts over the KV dimension, so the subgraph keeps working after the
/// rewriter shards it -- exactly like the ShapeOf-driven form real exports use.
inline std::shared_ptr<ov::Node> expand_kv_heads(const ov::Output<ov::Node>& kv, const BlockConfig& config) {
    const auto group = config.num_heads / config.num_kv_heads;

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2});
    auto unsqueezed = std::make_shared<ov::op::v0::Unsqueeze>(kv, axis);

    auto ones = ov::op::v0::Constant::create(ov::element::f32,
                                             ov::Shape{1, 1, group, 1, 1},
                                             std::vector<float>(group, 1.0f));
    auto expanded = std::make_shared<ov::op::v1::Multiply>(unsqueezed, ones);

    auto merged = ov::op::v0::Constant::create(
        ov::element::i64,
        ov::Shape{4},
        std::vector<int64_t>{0, static_cast<int64_t>(config.num_heads), -1, static_cast<int64_t>(config.head_dim)});
    return std::make_shared<ov::op::v1::Reshape>(expanded, merged, /*special_zero=*/true);
}

/// Wraps a projection output in a ReadValue/Assign KV cache, including the
/// initializer subgraph exported models carry:
/// `Constant(0) -> Broadcast(Concat([batch, kv_heads, 0, head_dim])) -> ReadValue`.
/// The KV head count in that Concat is one of the things the rewriter localizes.
inline std::shared_ptr<ov::Node> make_kv_cache(const ov::Output<ov::Node>& current,
                                               const std::string& variable_id,
                                               const BlockConfig& config,
                                               ov::SinkVector& sinks) {
    auto variable = std::make_shared<ov::op::util::Variable>(
        ov::op::util::VariableInfo{ov::PartialShape{1,
                                                    static_cast<int64_t>(config.num_kv_heads),
                                                    ov::Dimension::dynamic(),
                                                    static_cast<int64_t>(config.head_dim)},
                                   ov::element::f32,
                                   variable_id});

    auto zero = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{0.0f});
    auto batch = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    auto heads = ov::op::v0::Constant::create(ov::element::i64,
                                              ov::Shape{1},
                                              std::vector<int64_t>{static_cast<int64_t>(config.num_kv_heads)});
    auto length = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0});
    auto features = ov::op::v0::Constant::create(ov::element::i64,
                                                 ov::Shape{1},
                                                 std::vector<int64_t>{static_cast<int64_t>(config.head_dim)});
    auto target = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{batch, heads, length, features}, 0);
    auto init = std::make_shared<ov::op::v3::Broadcast>(zero, target);

    auto past = std::make_shared<ov::op::v6::ReadValue>(init, variable);
    auto present = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{past, current}, 2);
    sinks.push_back(std::make_shared<ov::op::v6::Assign>(present, variable));
    return present;
}

/// One transformer block: grouped-query attention followed by an MLP.
inline std::shared_ptr<ov::Model> make_transformer_block(const BlockConfig& config) {
    const auto q_features = config.num_heads * config.head_dim;
    const auto kv_features = config.num_kv_heads * config.head_dim;

    auto data = std::make_shared<ov::op::v0::Parameter>(
        ov::element::f32,
        ov::PartialShape{1, ov::Dimension::dynamic(), static_cast<int64_t>(config.hidden)});
    data->set_friendly_name("input");

    auto q = make_projection(data, config.hidden, q_features, config, config.with_bias, 1);
    auto k = make_projection(data, config.hidden, kv_features, config, config.with_bias, 2);
    auto v = make_projection(data, config.hidden, kv_features, config, config.with_bias, 3);

    auto q_heads = transpose_heads(reshape_to_heads(q, config.num_heads, config.head_dim));
    std::shared_ptr<ov::Node> k_heads = transpose_heads(reshape_to_heads(k, config.num_kv_heads, config.head_dim));
    std::shared_ptr<ov::Node> v_heads = transpose_heads(reshape_to_heads(v, config.num_kv_heads, config.head_dim));

    ov::SinkVector sinks;
    if (config.stateful) {
        k_heads = make_kv_cache(k_heads, "past_key_values.0.key", config, sinks);
        v_heads = make_kv_cache(v_heads, "past_key_values.0.value", config, sinks);
    }

    auto attention = std::make_shared<ov::op::v13::ScaledDotProductAttention>(q_heads,
                                                                             expand_kv_heads(k_heads, config),
                                                                             expand_kv_heads(v_heads, config),
                                                                             /*causal=*/true);

    auto merged = ov::op::v0::Constant::create(ov::element::i64,
                                               ov::Shape{3},
                                               std::vector<int64_t>{0, 0, static_cast<int64_t>(q_features)});
    auto attention_out =
        std::make_shared<ov::op::v1::Reshape>(transpose_heads(attention), merged, /*special_zero=*/true);

    auto out_proj = make_projection(attention_out, q_features, config.hidden, config, false, 4);
    auto residual = std::make_shared<ov::op::v1::Add>(data, out_proj);

    auto up = make_projection(residual, config.hidden, config.intermediate, config, false, 6);
    std::shared_ptr<ov::Node> mlp_body = std::make_shared<ov::op::v4::Swish>(up);
    if (config.gated_mlp) {
        auto gate = make_projection(residual, config.hidden, config.intermediate, config, false, 5);
        mlp_body = std::make_shared<ov::op::v1::Multiply>(std::make_shared<ov::op::v4::Swish>(gate), up);
    }
    auto down = make_projection(mlp_body, config.intermediate, config.hidden, config, false, 7);

    auto result = std::make_shared<ov::op::v1::Add>(residual, down);
    return std::make_shared<ov::Model>(ov::OutputVector{result}, sinks, ov::ParameterVector{data}, "TPTestBlock");
}

}  // namespace ov::tp_gpu::tests
