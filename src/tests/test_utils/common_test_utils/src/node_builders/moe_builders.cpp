// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "common_test_utils/node_builders/moe_builders.hpp"

#include <limits>
#include <memory>
#include <vector>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/core/model.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/broadcast.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/gather.hpp"
#include "openvino/op/gather_elements.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/shape_of.hpp"
#include "openvino/op/sigmoid.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/softmax.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "transformations/utils/utils.hpp"

namespace ov {
namespace test {

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>>
build_softmax_routing_subgraph(const ov::Output<ov::Node>& routing_weights, size_t number_of_experts, size_t topk) {
    using namespace ov::op;

    auto router_softmax = std::make_shared<v8::Softmax>(routing_weights, 1);

    auto k_const = v0::Constant::create(ov::element::i32, ov::Shape{}, {static_cast<int32_t>(topk)});
    auto router_topk =
        std::make_shared<v11::TopK>(router_softmax, k_const, 1, v11::TopK::Mode::MAX, v11::TopK::SortType::SORT_VALUES);

    auto reduce_sm =
        std::make_shared<v1::ReduceSum>(router_topk->output(0),
                                        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
                                        true);
    auto scatter_w = ov::Output<ov::Node>{std::make_shared<v1::Divide>(router_topk->output(0), reduce_sm)};
    auto topk_idx = router_topk->output(1);

    auto shapeof = std::make_shared<v3::ShapeOf>(topk_idx);
    auto gather = std::make_shared<v8::Gather>(shapeof,
                                               v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                               v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto unsq_seq =
        std::make_shared<v0::Unsqueeze>(gather,
                                        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}));

    auto ne_scalar = v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(number_of_experts)});
    auto unsq_ne =
        std::make_shared<v0::Unsqueeze>(ne_scalar,
                                        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}));

    ov::OutputVector bcast_vec{unsq_seq, unsq_ne};
    auto bcast_shape = std::make_shared<v0::Concat>(bcast_vec, 0);

    auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    ov::OutputVector reshape_vec{unsq_ne, unsq_seq, one};
    auto reshape_shape = std::make_shared<v0::Concat>(reshape_vec, 0);

    auto zero = v0::Constant::create(scatter_w.get_element_type(), ov::Shape{1}, {0});
    auto bcast = std::make_shared<v3::Broadcast>(zero, bcast_shape);
    auto scatter = std::make_shared<v12::ScatterElementsUpdate>(
        bcast,
        topk_idx,
        scatter_w,
        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
    auto transp = std::make_shared<v1::Transpose>(
        scatter,
        v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));
    auto reshape = std::make_shared<v1::Reshape>(transp, reshape_shape, false);
    auto unsqueeze_moe = ov::Output<ov::Node>{
        std::make_shared<v0::Unsqueeze>(reshape,
                                        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3}))};

    return {unsqueeze_moe, topk_idx};
}

std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> build_sigmoid_bias_routing_subgraph(
    const ov::Output<ov::Node>& routing_weights,
    ov::element::Type data_precision,
    size_t number_of_experts,
    size_t topk) {
    using namespace ov::op;

    auto sigmoid = std::make_shared<v0::Sigmoid>(routing_weights);
    auto bias_test_data = ov::test::utils::InputGenerateData(0., 1, 100);
    auto bias = ov::test::utils::make_constant(data_precision, ov::Shape{1, number_of_experts}, bias_test_data);
    auto sig_add = std::make_shared<v1::Add>(sigmoid, bias);

    auto router_topk =
        std::make_shared<v11::TopK>(sig_add,
                                    v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(topk)}),
                                    -1,
                                    v11::TopK::Mode::MAX,
                                    v11::TopK::SortType::SORT_VALUES,
                                    ov::element::i64);

    auto convert_topk = std::make_shared<v0::Convert>(router_topk->output(1), ov::element::i32);
    auto topk_idx = ov::Output<ov::Node>{convert_topk};

    auto gather_el = std::make_shared<v6::GatherElements>(sigmoid, convert_topk, 1);
    auto reduce =
        std::make_shared<v1::ReduceSum>(gather_el,
                                        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
                                        true);
    auto eps = v0::Constant::create(data_precision, ov::Shape{1, 1}, {1e-6f});
    auto add_eps = std::make_shared<v1::Add>(reduce, eps);
    auto norm = std::make_shared<v1::Divide>(gather_el, add_eps);

    auto sl_stop = std::make_shared<v3::ShapeOf>(convert_topk, ov::element::i32);
    auto scatter_w = ov::Output<ov::Node>{
        std::make_shared<v8::Slice>(norm,
                                    v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{0, 0}),
                                    sl_stop,
                                    v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 1}),
                                    v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{0, 1}))};

    auto shapeof = std::make_shared<v3::ShapeOf>(convert_topk, ov::element::i64);
    auto gather = std::make_shared<v8::Gather>(shapeof,
                                               v0::Constant::create(ov::element::i64, ov::Shape{}, {0}),
                                               v0::Constant::create(ov::element::i64, ov::Shape{}, {0}));
    auto unsq_seq =
        std::make_shared<v0::Unsqueeze>(gather,
                                        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}));

    auto ne_scalar = v0::Constant::create(ov::element::i64, ov::Shape{}, {static_cast<int64_t>(number_of_experts)});
    auto unsq_ne =
        std::make_shared<v0::Unsqueeze>(ne_scalar,
                                        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}));

    ov::OutputVector bcast_vec{unsq_seq, unsq_ne};
    auto bcast_shape = std::make_shared<v0::Concat>(bcast_vec, 0);

    auto zero = v0::Constant::create(scatter_w.get_element_type(), ov::Shape{1}, {0});
    auto bcast = std::make_shared<v3::Broadcast>(zero, bcast_shape);
    auto scatter = std::make_shared<v12::ScatterElementsUpdate>(
        bcast,
        topk_idx,
        scatter_w,
        v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1}));
    auto transp = std::make_shared<v1::Transpose>(
        scatter,
        v0::Constant::create(ov::element::i32, ov::Shape{2}, std::vector<int32_t>{1, 0}));

    auto one = v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
    ov::OutputVector reshape_vec{unsq_ne, one, unsq_seq};
    auto reshape_shape = std::make_shared<v0::Concat>(reshape_vec, 0);
    auto reshape = std::make_shared<v1::Reshape>(transp, reshape_shape, false);

    auto unsqueeze_moe = ov::Output<ov::Node>{
        std::make_shared<v0::Unsqueeze>(reshape,
                                        v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3}))};

    return {unsqueeze_moe, topk_idx};
}

std::shared_ptr<ov::Node> build_matmul_weights(
    const ov::Shape& weights_shape,
    const ov::element::Type& weights_precision,
    const ov::element::Type& data_precision,
    size_t seed,
    bool use_weight_decompression,
    const std::optional<ov::element::Type>& decompression_precision = std::nullopt,
    const std::optional<ov::element::Type>& scale_precision = std::nullopt,
    const std::optional<ov::test::utils::DecompressionType>& decompression_multiply_type = std::nullopt,
    const std::optional<ov::test::utils::DecompressionType>& decompression_subtract_type = std::nullopt,
    const std::optional<bool>& reshape_on_decompression = std::nullopt,
    const std::optional<int>& decompression_group_size = std::nullopt) {
    if (!use_weight_decompression) {
        // Note: builder takes planar weights shape, but it is transposed here
        // since MoE matmuls are supported only in case of transpose_b=true
        auto transposed_weights_shape = weights_shape;
        std::swap(*transposed_weights_shape.rbegin(), *(transposed_weights_shape.rbegin() + 1));
        std::shared_ptr<ov::Node> weights = ov::test::utils::make_constant(weights_precision,
                                                                           transposed_weights_shape,
                                                                           utils::InputGenerateData(0, 10, 1, seed));
        if (weights_precision != data_precision) {
            weights = std::make_shared<ov::op::v0::Convert>(weights, data_precision);
        }
        return weights;
    } else {
        OPENVINO_ASSERT(decompression_precision.has_value(),
                        "decompression_precision must be set when use_weight_decompression is true");
        OPENVINO_ASSERT(scale_precision.has_value(),
                        "scale_precision must be set when use_weight_decompression is true");
        OPENVINO_ASSERT(decompression_multiply_type.has_value(),
                        "decompression_multiply_type must be set when use_weight_decompression is true");
        OPENVINO_ASSERT(decompression_subtract_type.has_value(),
                        "decompression_subtract_type must be set when use_weight_decompression is true");
        OPENVINO_ASSERT(reshape_on_decompression.has_value(),
                        "reshape_on_decompression must be set when use_weight_decompression is true");
        OPENVINO_ASSERT(decompression_group_size.has_value(),
                        "decompression_group_size must be set when use_weight_decompression is true");
        return ov::test::utils::initMatMulDecompressionSubgraphQuantization(weights_shape,
                                                                            decompression_group_size.value(),
                                                                            data_precision,
                                                                            weights_precision,
                                                                            decompression_precision.value(),
                                                                            scale_precision.value(),
                                                                            true,
                                                                            decompression_multiply_type.value(),
                                                                            decompression_subtract_type.value(),
                                                                            reshape_on_decompression.value(),
                                                                            false,
                                                                            seed);
    }
}

std::shared_ptr<ov::Model> initMoE2GeMMSubgraph(
    const MoePatternParams& moe_params,
    const ov::element::Type data_precision,
    const ov::element::Type weights_precision,
    const bool use_weight_decompression,
    const std::optional<ov::element::Type> decompression_precision,
    const std::optional<ov::element::Type> scale_precision,
    const std::optional<ov::test::utils::DecompressionType> decompression_multiply_type,
    const std::optional<ov::test::utils::DecompressionType> decompression_subtract_type,
    const std::optional<bool> reshape_on_decompression,
    const std::optional<int> decompression_group_size) {
    // Use parameters from shape_params - static shapes only
    const auto& input_shape = moe_params.data_shape;
    const size_t intermediate_size = moe_params.intermediate_size;
    const size_t topk = moe_params.topk;
    const size_t number_of_experts = moe_params.number_of_experts;

    const auto expert_alpha = 1.625f;
    const auto expert_beta = 7.0f;

    constexpr int64_t fusion_factor = 2;  // property of GPT-OSS
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

    // Note: we need to use different seed to avoid the exact weights generation for the MatMuls with the same shape
    size_t seed = 1;
    auto gate_up_weights =
        build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size * fusion_factor},
                             weights_precision,
                             data_precision,
                             seed++,
                             use_weight_decompression,
                             decompression_precision,
                             scale_precision,
                             decompression_multiply_type,
                             decompression_subtract_type,
                             reshape_on_decompression,
                             decompression_group_size);

    auto gate_up_matmul = std::make_shared<ov::op::v0::MatMul>(after_tile_reshape, gate_up_weights, false, true);

    gate_up_matmul->set_friendly_name("GateUpMatMul");

    auto gate_up_add = std::make_shared<ov::op::v1::Add>(
        gate_up_matmul,
        ov::test::utils::make_constant(data_precision,
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
    auto add1 = std::make_shared<ov::op::v1::Add>(clamp, ov::test::utils::make_constant(data_precision, ov::Shape{1}));

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
                                              ov::op::v0::Constant::create(data_precision, ov::Shape{1}, {10.0f}));
    auto swish_beta = ov::op::v0::Constant::create(data_precision, ov::Shape{}, std::vector<float>{expert_alpha});
    auto swish = std::make_shared<ov::op::v4::Swish>(minimum1, swish_beta);

    auto multiply2 = std::make_shared<ov::op::v1::Multiply>(add1, swish);

    // Down projection
    auto down_proj_weights = build_matmul_weights(ov::Shape{number_of_experts, intermediate_size, hidden_size},
                                                  weights_precision,
                                                  data_precision,
                                                  seed++,
                                                  use_weight_decompression,
                                                  decompression_precision,
                                                  scale_precision,
                                                  decompression_multiply_type,
                                                  decompression_subtract_type,
                                                  reshape_on_decompression,
                                                  decompression_group_size);

    auto down_proj_matmul = std::make_shared<ov::op::v0::MatMul>(multiply2, down_proj_weights, false, true);

    down_proj_matmul->set_friendly_name("DownProjMatMul");

    auto down_proj_add = std::make_shared<ov::op::v1::Add>(
        down_proj_matmul,
        ov::test::utils::make_constant(data_precision, ov::Shape{number_of_experts, 1, hidden_size}));

    auto router_weights = build_matmul_weights(ov::Shape{hidden_size, number_of_experts},
                                               weights_precision,
                                               data_precision,
                                               seed++,
                                               use_weight_decompression,
                                               decompression_precision,
                                               scale_precision,
                                               decompression_multiply_type,
                                               decompression_subtract_type,
                                               reshape_on_decompression,
                                               decompression_group_size);

    auto reshape_2nd_consumer_router_matmul =
        std::make_shared<ov::op::v0::MatMul>(experts_reshape, router_weights, false, true);

    auto router_bias = std::make_shared<ov::op::v1::Add>(
        reshape_2nd_consumer_router_matmul,
        ov::test::utils::make_constant(data_precision, ov::Shape{1, number_of_experts}));

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

    auto shape_of = std::make_shared<ov::op::v0::ShapeOf>(router_bias);
    auto zero_const = ov::op::v0::Constant::create(slice3->get_output_element_type(0), ov::Shape{1}, {0});
    auto broadcast_zero = std::make_shared<ov::op::v3::Broadcast>(zero_const, shape_of);

    auto scatter_elements_update = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
        broadcast_zero,
        router_topk_indices,
        slice3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));

    auto router_transpose = std::make_shared<ov::op::v1::Transpose>(
        scatter_elements_update,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));

    const auto number_of_experts_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(number_of_experts)});
    auto first_in_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(input, {0});
    auto last_in_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(input, {2});
    auto minus_one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    const auto router_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{number_of_experts_const, first_in_dim, minus_one}, 0);

    auto router_reshape = std::make_shared<ov::op::v1::Reshape>(router_transpose, router_shape, true);

    auto unsqueeze_routing_weights = std::make_shared<ov::op::v0::Unsqueeze>(
        router_reshape,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3}));

    auto end_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{number_of_experts_const, first_in_dim, minus_one, last_in_dim},
        0);
    auto end_reshape = std::make_shared<ov::op::v1::Reshape>(down_proj_add, end_shape, true);

    auto mul3 = std::make_shared<ov::op::v1::Multiply>(end_reshape, unsqueeze_routing_weights);

    // ReduceSum - final node of the MOE pattern
    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        false);

    ov::ParameterVector params = {input};
    return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(reduce_sum)},
                                       params,
                                       "MoeSubgraph");
}

std::shared_ptr<ov::Model> initMoE3GeMMSubgraph(
    const MoePatternParams& moe_params,
    const ov::element::Type data_precision,
    const ov::element::Type weights_precision,
    const bool use_weight_decompression,
    const std::optional<ov::element::Type> decompression_precision,
    const std::optional<ov::element::Type> scale_precision,
    const std::optional<ov::test::utils::DecompressionType> decompression_multiply_type,
    const std::optional<ov::test::utils::DecompressionType> decompression_subtract_type,
    const std::optional<bool> reshape_on_decompression,
    const std::optional<int> decompression_group_size,
    MoERoutingType routing_type) {
    // Use parameters from shape_params - static shapes only
    const auto& input_shape = moe_params.data_shape;
    const size_t intermediate_size = moe_params.intermediate_size;
    const size_t topk = moe_params.topk;
    const size_t number_of_experts = moe_params.number_of_experts;

    const auto expert_alpha = 1.702f;
    const auto input_rank = static_cast<size_t>(input_shape.rank().get_length());
    OPENVINO_ASSERT(input_rank >= 2 && input_shape[input_rank - 1].is_static());
    const auto hidden_size = static_cast<size_t>(input_shape[input_rank - 1].get_length());

    // Create input parameter
    auto input = std::make_shared<ov::op::v0::Parameter>(data_precision, input_shape);
    ov::Output<ov::Node> input_data = input;

    // Expert processing path - use -1 for dynamic reshape like in reference
    auto experts_reshape = std::make_shared<ov::op::v1::Reshape>(
        input_data,
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

    // Note: we need to use different seed to avoid the exact weights generation for the MatMuls with the same shape
    size_t seed = 1;
    auto gate_weights = build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size},
                                             weights_precision,
                                             data_precision,
                                             seed++,
                                             use_weight_decompression,
                                             decompression_precision,
                                             scale_precision,
                                             decompression_multiply_type,
                                             decompression_subtract_type,
                                             reshape_on_decompression,
                                             decompression_group_size);

    auto gate_matmul = std::make_shared<ov::op::v0::MatMul>(after_tile_reshape, gate_weights, false, true);

    gate_matmul->set_friendly_name("GateMatMul");

    // Apply Swish activation directly to gate
    auto swish = std::make_shared<ov::op::v4::Swish>(gate_matmul);

    // Second GEMM (up_projection)
    auto up_weights = build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size},
                                           weights_precision,
                                           data_precision,
                                           seed++,
                                           use_weight_decompression,
                                           decompression_precision,
                                           scale_precision,
                                           decompression_multiply_type,
                                           decompression_subtract_type,
                                           reshape_on_decompression,
                                           decompression_group_size);

    auto up_matmul = std::make_shared<ov::op::v0::MatMul>(after_tile_reshape, up_weights, false, true);

    up_matmul->set_friendly_name("UpMatMul");

    // Join: Multiply (SwiGLU)
    auto swiglu = std::make_shared<ov::op::v1::Multiply>(swish, up_matmul);

    // Third GEMM (down_projection)
    auto down_weights_moe3 = build_matmul_weights(ov::Shape{number_of_experts, intermediate_size, hidden_size},
                                                  weights_precision,
                                                  data_precision,
                                                  seed++,
                                                  use_weight_decompression,
                                                  decompression_precision,
                                                  scale_precision,
                                                  decompression_multiply_type,
                                                  decompression_subtract_type,
                                                  reshape_on_decompression,
                                                  decompression_group_size);

    auto down_matmul = std::make_shared<ov::op::v0::MatMul>(swiglu, down_weights_moe3, false, true);

    down_matmul->set_friendly_name("DownMatMul");

    // Router subgraph - this is crucial for the MoE pattern recognition
    auto router_weights_moe3 = build_matmul_weights(ov::Shape{hidden_size, number_of_experts},
                                                    weights_precision,
                                                    data_precision,
                                                    seed++,
                                                    use_weight_decompression,
                                                    decompression_precision,
                                                    scale_precision,
                                                    decompression_multiply_type,
                                                    decompression_subtract_type,
                                                    reshape_on_decompression,
                                                    decompression_group_size);

    auto router_matmul = std::make_shared<ov::op::v0::MatMul>(experts_reshape, router_weights_moe3, false, true);

    std::pair<ov::Output<ov::Node>, ov::Output<ov::Node>> routing_outputs;
    switch (routing_type) {
    case MoERoutingType::SOFTMAX:
        routing_outputs = build_softmax_routing_subgraph(router_matmul, number_of_experts, topk);
        break;
    case MoERoutingType::SIGMOID_BIAS:
        routing_outputs = build_sigmoid_bias_routing_subgraph(router_matmul, data_precision, number_of_experts, topk);
        break;
    default:
        OPENVINO_THROW("Unsupported MoERoutingType");
    }
    auto [unsqueeze_routing_weights, router_topk_indices] = routing_outputs;

    const auto number_of_experts_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(number_of_experts)});
    auto minus_one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    auto one_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});

    auto first_topk_dim_for_reshape =
        ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(router_topk_indices, {0});

    auto last_in_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(input, {input_rank - 1});

    // Softmax: end_reshape [ne, seq, 1, hidden]  matched by mul with [ne, seq, 1, 1]
    // Sigmoid: end_reshape [ne, 1, seq, hidden]  matched by mul with [ne, 1, seq, 1]
    std::shared_ptr<ov::op::v0::Concat> end_shape;
    if (routing_type == MoERoutingType::SOFTMAX) {
        ov::OutputVector end_dims{number_of_experts_const, first_topk_dim_for_reshape, minus_one, last_in_dim};
        end_shape = std::make_shared<ov::op::v0::Concat>(end_dims, 0);
    } else {
        ov::OutputVector end_dims{number_of_experts_const, one_const, first_topk_dim_for_reshape, last_in_dim};
        end_shape = std::make_shared<ov::op::v0::Concat>(end_dims, 0);
    }
    auto end_reshape = std::make_shared<ov::op::v1::Reshape>(down_matmul, end_shape, true);

    auto mul3 = std::make_shared<ov::op::v1::Multiply>(end_reshape, unsqueeze_routing_weights);

    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        false);

    // Flatten any leading dimensions to [batch*seq, hidden] regardless of routing type
    auto final_reshape_const =
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)});
    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(reduce_sum, final_reshape_const, true);

    ov::ParameterVector params = {input};
    return std::make_shared<ov::Model>(ov::OutputVector{std::make_shared<ov::op::v0::Result>(final_reshape)},
                                       params,
                                       "MoE3GeMMSubgraph");
}

}  // namespace test
}  // namespace ov