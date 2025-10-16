// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "shared_test_classes/subgraph/moe_builders.hpp"

#include <limits>
#include <memory>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"

namespace ov {
namespace test {

std::shared_ptr<ov::op::v1::ReduceSum> ApplyExpertsSubgraph(const ov::Output<ov::Node>& batched_mm_output,
                                                            const ov::Output<ov::Node>& scatter_bcast_shape,
                                                            const ov::Output<ov::Node>& router_topk_indices,
                                                            const ov::Output<ov::Node>& routing_values,
                                                            const ov::Output<ov::Node>& model_input,
                                                            const size_t number_of_experts) {
    const auto fetch_input_shape = std::make_shared<ov::op::v0::ShapeOf>(model_input);
    const auto number_of_experts_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(number_of_experts)});

    // Create broadcast_zero - common to both builders
    auto zero_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {0});
    auto broadcast_zero = std::make_shared<ov::op::v3::Broadcast>(zero_const, scatter_bcast_shape);

    // Common routing weights processing
    auto scatter_elements_update = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
        broadcast_zero,
        router_topk_indices,
        routing_values,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));  // axis

    // Transpose: Dynamic shape transpose
    auto router_transpose = std::make_shared<ov::op::v1::Transpose>(
        scatter_elements_update,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));

    const auto batch_seq_len = std::make_shared<ov::op::v1::Gather>(
        fetch_input_shape,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {0}));

    const auto router_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{number_of_experts_const, batch_seq_len}, 0);

    auto router_reshape = std::make_shared<ov::op::v1::Reshape>(router_transpose, router_shape, true);

    auto unsqueeze_routing_weights = std::make_shared<ov::op::v0::Unsqueeze>(
        router_reshape,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}));

    // Apply routing weights to expert output
    const auto end_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{number_of_experts_const, fetch_input_shape}, 0);
    auto end_reshape = std::make_shared<ov::op::v1::Reshape>(batched_mm_output, end_shape, false);

    auto mul3 = std::make_shared<ov::op::v1::Multiply>(end_reshape, unsqueeze_routing_weights);

    // ReduceSum - final node of the MOE pattern
    return std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        false);
}

std::shared_ptr<ov::Model> initMoE2GeMMSubgraph(const MoePatternParams& moe_params,
                                                const ov::element::Type data_precision,
                                                const ov::element::Type weights_precision,
                                                const ov::element::Type decompression_precision,
                                                const ov::element::Type scale_precision,
                                                const bool use_weight_decompression,
                                                const DecompressionType decompression_multiply_type,
                                                const DecompressionType decompression_subtract_type,
                                                const bool reshape_on_decompression,
                                                const int decompression_group_size) {
    // Use parameters from shape_params - static shapes only
    const auto& data_shape = moe_params.data_shape;
    const size_t intermediate_size = moe_params.intermediate_size;
    const size_t topk = moe_params.topk;
    const size_t number_of_experts = moe_params.number_of_experts;

    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    constexpr int64_t fusion_factor = 2;  // property of GPT-OSS

    const auto& input_shape = data_shape.first;
    // Create input parameter with dynamic shape - batch and hidden_size are fixed, seq_len is dynamic
    auto input = std::make_shared<ov::op::v0::Parameter>(data_precision, input_shape);

    OPENVINO_ASSERT(input_shape[2].is_static());
    const auto hidden_size = static_cast<size_t>(input_shape[2].get_length());

    // Expert processing path - use -1 for dynamic reshape like in reference
    auto experts_reshape = std::make_shared<ov::op::v1::Reshape>(
        input,
        ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{2},
            std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),  // -1 flattens batch*seq_len
        false);

    auto tile = std::make_shared<ov::op::v0::Tile>(
        experts_reshape,
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     std::vector<int64_t>{static_cast<int64_t>(number_of_experts), 1}));

    auto after_tile_reshape = std::make_shared<ov::op::v1::Reshape>(
        tile,
        ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(number_of_experts), -1, static_cast<int64_t>(hidden_size)}),
        false);

    // Gate/Up projection
    auto gate_up_matmul = std::make_shared<ov::op::v0::MatMul>(
        after_tile_reshape,
        ov::test::utils::make_constant(ov::element::f32,
                                       ov::Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size}),
        false,
        true);

    gate_up_matmul->set_friendly_name("GateUpMatMul");

    auto gate_up_add = std::make_shared<ov::op::v1::Add>(
        gate_up_matmul,
        ov::test::utils::make_constant(ov::element::f32,
                                       ov::Shape{number_of_experts, 1, intermediate_size * fusion_factor}));

    // Slice the last axis, every second element
    auto slice1 = std::make_shared<ov::op::v8::Slice>(
        gate_up_add,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{1},
                                     std::vector<int64_t>{std::numeric_limits<int64_t>::max()}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}));
    auto clamp = std::make_shared<ov::op::v0::Clamp>(slice1, -expert_beta, expert_beta);
    auto add1 =
        std::make_shared<ov::op::v1::Add>(clamp, ov::test::utils::make_constant(ov::element::f32, ov::Shape{1}));

    auto slice2 = std::make_shared<ov::op::v8::Slice>(
        gate_up_add,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{1},
                                     std::vector<int64_t>{std::numeric_limits<int64_t>::max()}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}));
    auto minimum1 =
        std::make_shared<ov::op::v1::Minimum>(slice2,
                                              ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1}, {10.0f}));
    auto swish_beta = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, std::vector<float>{expert_alpha});
    auto swish = std::make_shared<ov::op::v4::Swish>(minimum1, swish_beta);

    auto multiply2 = std::make_shared<ov::op::v1::Multiply>(add1, swish);

    // Down projection
    auto down_proj_matmul = std::make_shared<ov::op::v0::MatMul>(
        multiply2,
        ov::test::utils::make_constant(ov::element::f32, ov::Shape{number_of_experts, hidden_size, intermediate_size}),
        false,
        true);

    down_proj_matmul->set_friendly_name("DownProjMatMul");

    auto down_proj_add = std::make_shared<ov::op::v1::Add>(
        down_proj_matmul,
        ov::test::utils::make_constant(ov::element::f32, ov::Shape{number_of_experts, 1, hidden_size}));

    // Router subgraph - this is crucial for the MoE pattern recognition
    auto reshape_2nd_consumer_router_matmul = std::make_shared<ov::op::v0::MatMul>(
        experts_reshape,
        ov::test::utils::make_constant(ov::element::f32, ov::Shape{number_of_experts, hidden_size}),
        false,
        true);

    auto router_bias = std::make_shared<ov::op::v1::Add>(
        reshape_2nd_consumer_router_matmul,
        ov::test::utils::make_constant(ov::element::f32, ov::Shape{1, number_of_experts}));

    auto router_topk_values_and_indices =
        std::make_shared<ov::op::v11::TopK>(router_bias,
                                            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {topk}),
                                            -1,
                                            ov::op::v11::TopK::Mode::MAX,
                                            ov::op::v11::TopK::SortType::SORT_VALUES,
                                            ov::element::i64);

    auto router_topk_values = router_topk_values_and_indices->output(0);
    auto router_topk_values_softmax = std::make_shared<ov::op::v1::Softmax>(router_topk_values, 1);
    auto router_topk_indices = router_topk_values_and_indices->output(1);
    auto slice3 = std::make_shared<ov::op::v8::Slice>(
        router_topk_values_softmax,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 0}),
        std::make_shared<ov::op::v3::ShapeOf>(router_topk_indices),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 1}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1}));

    // add ShapeOf after Add
    auto shape_of = std::make_shared<ov::op::v0::ShapeOf>(router_bias);

    auto reduce_sum =
        ApplyExpertsSubgraph(down_proj_add, shape_of, router_topk_indices, slice3, input, number_of_experts);

    ov::ParameterVector params = {input};
    return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(reduce_sum)},
                                       params,
                                       "MoeSubgraph");
}

std::shared_ptr<ov::Model> initMoE3GeMMSubgraph(const MoePatternParams& moe_params,
                                                const ov::element::Type data_precision,
                                                const ov::element::Type weights_precision,
                                                const ov::element::Type decompression_precision,
                                                const ov::element::Type scale_precision,
                                                const bool use_weight_decompression,
                                                const DecompressionType decompression_multiply_type,
                                                const DecompressionType decompression_subtract_type,
                                                const bool reshape_on_decompression,
                                                const int decompression_group_size) {
    // Use parameters from shape_params - static shapes only
    const auto& data_shape = moe_params.data_shape;
    const size_t intermediate_size = moe_params.intermediate_size;
    const size_t topk = moe_params.topk;
    const size_t number_of_experts = moe_params.number_of_experts;

    const auto expert_alpha = 1.702f;

    const auto& input_shape = data_shape.first;
    // Create input parameter with dynamic shape - batch and hidden_size are fixed, seq_len is dynamic
    auto input = std::make_shared<ov::op::v0::Parameter>(data_precision, input_shape);

    OPENVINO_ASSERT(input_shape[2].is_static());
    const auto hidden_size = static_cast<size_t>(input_shape[2].get_length());

    // Expert processing path - use -1 for dynamic reshape like in reference
    auto experts_reshape = std::make_shared<ov::op::v1::Reshape>(
        input,
        ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{2},
            std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),  // -1 flattens batch*seq_len
        false);

    auto tile = std::make_shared<ov::op::v0::Tile>(
        experts_reshape,
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     std::vector<int64_t>{static_cast<int64_t>(number_of_experts), 1}));

    auto after_tile_reshape = std::make_shared<ov::op::v1::Reshape>(
        tile,
        ov::op::v0::Constant::create(
            ov::element::i64,
            ov::Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(number_of_experts), -1, static_cast<int64_t>(hidden_size)}),
        false);

    // First GEMM (activation gate)
    auto gate_matmul = std::make_shared<ov::op::v0::MatMul>(
        after_tile_reshape,
        ov::test::utils::make_constant(ov::element::f32,
                                       ov::Shape{number_of_experts, intermediate_size, hidden_size},
                                       utils::InputGenerateData(0, 10, 1, 1)),
        false,
        true);

    gate_matmul->set_friendly_name("GateMatMul");

    // Apply Swish activation directly to gate
    auto swish = std::make_shared<ov::op::v4::Swish>(gate_matmul);

    // Second GEMM (up_projection)
    auto up_matmul = std::make_shared<ov::op::v0::MatMul>(
        after_tile_reshape,
        ov::test::utils::make_constant(ov::element::f32,
                                       ov::Shape{number_of_experts, intermediate_size, hidden_size},
                                       utils::InputGenerateData(0, 10, 1, 2)),
        false,
        true);

    up_matmul->set_friendly_name("UpMatMul");

    // Join: Multiply (SwiGLU)
    auto swiglu = std::make_shared<ov::op::v1::Multiply>(swish, up_matmul);

    // Third GEMM (down_projection)
    auto down_matmul = std::make_shared<ov::op::v0::MatMul>(
        swiglu,
        ov::test::utils::make_constant(ov::element::f32, ov::Shape{number_of_experts, hidden_size, intermediate_size}),
        false,
        true);

    down_matmul->set_friendly_name("DownMatMul");

    // Router subgraph - this is crucial for the MoE pattern recognition
    auto reshape_2nd_consumer_router_matmul = std::make_shared<ov::op::v0::MatMul>(
        experts_reshape,
        ov::test::utils::make_constant(ov::element::f32, ov::Shape{number_of_experts, hidden_size}),
        false,
        true);

    auto router_softmax = std::make_shared<ov::op::v1::Softmax>(reshape_2nd_consumer_router_matmul, 1);

    auto router_topk_values_and_indices =
        std::make_shared<ov::op::v11::TopK>(router_softmax,
                                            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {topk}),
                                            -1,
                                            ov::op::v11::TopK::Mode::MAX,
                                            ov::op::v11::TopK::SortType::SORT_VALUES,
                                            ov::element::i64);

    auto router_topk_values_reduce = std::make_shared<ov::op::v1::ReduceSum>(
        router_topk_values_and_indices->output(0),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}),
        false);
    auto router_topk_values_normalization =
        std::make_shared<ov::op::v1::Divide>(router_topk_values_and_indices->output(0), router_topk_values_reduce);
    auto router_topk_indices = router_topk_values_and_indices->output(1);

    auto shape_of = std::make_shared<ov::op::v0::ShapeOf>(router_topk_values_and_indices->output(1));
    auto gather = std::make_shared<ov::op::v8::Gather>(
        shape_of,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{0}));
    auto concat = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{gather,
                         ov::op::v0::Constant::create(ov::element::i64,
                                                      ov::Shape{1},
                                                      std::vector<int64_t>{static_cast<int64_t>(number_of_experts)})},
        0);

    auto reduce_sum = ApplyExpertsSubgraph(down_matmul,
                                           concat,
                                           router_topk_indices,
                                           router_topk_values_normalization,
                                           input,
                                           number_of_experts);

    ov::ParameterVector params = {input};
    return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(reduce_sum)},
                                       params,
                                       "MoE3GeMMSubgraph");
}

}  // namespace test
}  // namespace ov