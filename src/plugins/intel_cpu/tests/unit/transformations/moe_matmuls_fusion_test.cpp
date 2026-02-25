// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/moe_matmuls_fusion.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "common_test_utils/subgraph_builders/weights_decompression_builders.hpp"
#include "openvino/core/model.hpp"
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
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/common/op/batch_gather_matmul.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

namespace {

enum class MoEType { MoE2GeMM, MoE3GeMM };

inline std::ostream& operator<<(std::ostream& os, const MoEType& type) {
    switch (type) {
    case MoEType::MoE2GeMM:
        return os << "MoE2GeMM";
    case MoEType::MoE3GeMM:
        return os << "MoE3GeMM";
    default:
        OPENVINO_THROW("Unsupported MoEType");
    }
}

using MoEMatMulsFusionParams = std::tuple<MoEType,  // moe_type
                                          bool,     // use_scatter_v12
                                          bool,     // use_broadcast_v3
                                          bool,     // skip_unsqueeze
                                          bool,     // matmul_transpose_b
                                          bool,     // additional_node_consumers
                                          bool>;    // transformation_should_be_applied

inline std::shared_ptr<ov::Node> build_matmul_weights(const ov::Shape& weights_shape,
                                                      const ov::element::Type& weights_precision,
                                                      size_t seed,
                                                      bool transpose_b) {
    // Note: builder takes planar weights shape, but it is transposed here
    // only if transpose_b=true (which is the default case for MoE matmuls)
    auto final_weights_shape = weights_shape;
    if (transpose_b) {
        std::swap(*final_weights_shape.rbegin(), *(final_weights_shape.rbegin() + 1));
    }
    return ov::test::utils::make_constant(weights_precision,
                                          final_weights_shape,
                                          ov::test::utils::InputGenerateData(0, 10, 1, seed));
}

inline std::shared_ptr<ov::Model> initMoE2GeMMSubgraph(bool use_scatter_v12,
                                                       bool use_broadcast_v3,
                                                       bool skip_unsqueeze,
                                                       bool matmul_transpose_b,
                                                       bool additional_node_consumers) {
    // Fixed values that don't affect pass behavior
    const auto expert_alpha = 1.625f;
    const auto expert_beta = 7.0f;
    const ov::PartialShape input_shape = {-1, -1, 256};
    const size_t topk = 4;
    const size_t number_of_experts = 8;
    const size_t intermediate_size = 512;
    const ov::element::Type data_precision = ov::element::f32;
    const ov::element::Type weights_precision = ov::element::f32;
    constexpr int64_t fusion_factor = 2;

    auto input = std::make_shared<ov::op::v0::Parameter>(data_precision, input_shape);
    const auto hidden_size = static_cast<size_t>(input_shape[2].get_length());

    auto experts_reshape = std::make_shared<ov::op::v1::Reshape>(
        input,
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
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
                             seed++,
                             matmul_transpose_b);

    auto gate_up_matmul =
        std::make_shared<ov::op::v0::MatMul>(after_tile_reshape, gate_up_weights, false, matmul_transpose_b);

    gate_up_matmul->set_friendly_name("GateUpMatMul");

    auto gate_up_add = std::make_shared<ov::op::v1::Add>(
        gate_up_matmul,
        ov::test::utils::make_constant(data_precision,
                                       ov::Shape{number_of_experts, 1, intermediate_size * fusion_factor}));

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

    auto down_proj_weights = build_matmul_weights(ov::Shape{number_of_experts, intermediate_size, hidden_size},
                                                  weights_precision,
                                                  seed++,
                                                  matmul_transpose_b);

    auto down_proj_matmul =
        std::make_shared<ov::op::v0::MatMul>(multiply2, down_proj_weights, false, matmul_transpose_b);

    down_proj_matmul->set_friendly_name("DownProjMatMul");

    auto down_proj_add = std::make_shared<ov::op::v1::Add>(
        down_proj_matmul,
        ov::test::utils::make_constant(data_precision, ov::Shape{number_of_experts, 1, hidden_size}));

    auto router_weights =
        build_matmul_weights(ov::Shape{hidden_size, number_of_experts}, weights_precision, seed++, matmul_transpose_b);

    auto reshape_2nd_consumer_router_matmul =
        std::make_shared<ov::op::v0::MatMul>(experts_reshape, router_weights, false, matmul_transpose_b);

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

    std::shared_ptr<ov::Node> broadcast_zero;
    if (use_broadcast_v3) {
        broadcast_zero = std::make_shared<ov::op::v3::Broadcast>(zero_const, shape_of);
    } else {
        broadcast_zero = std::make_shared<ov::op::v1::Broadcast>(zero_const, shape_of);
    }

    std::shared_ptr<ov::Node> scatter_elements_update;
    if (use_scatter_v12) {
        scatter_elements_update = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
            broadcast_zero,
            router_topk_indices,
            slice3,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
    } else {
        scatter_elements_update = std::make_shared<ov::op::v3::ScatterElementsUpdate>(
            broadcast_zero,
            router_topk_indices,
            slice3,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
    }

    auto router_transpose = std::make_shared<ov::op::v1::Transpose>(
        scatter_elements_update,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));

    const auto number_of_experts_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(number_of_experts)});
    auto first_in_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(input, {0});
    auto last_in_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(input, {2});
    auto minus_one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    std::shared_ptr<ov::op::v0::Concat> router_shape;
    if (skip_unsqueeze) {
        auto one_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        router_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{number_of_experts_const, first_in_dim, minus_one, one_const},
            0);
    } else {
        router_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{number_of_experts_const, first_in_dim, minus_one}, 0);
    }

    auto router_reshape = std::make_shared<ov::op::v1::Reshape>(router_transpose, router_shape, true);

    std::shared_ptr<ov::Node> routing_weights_final;
    if (skip_unsqueeze) {
        routing_weights_final = router_reshape;
    } else {
        auto unsqueeze_routing_weights = std::make_shared<ov::op::v0::Unsqueeze>(
            router_reshape,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3}));
        routing_weights_final = unsqueeze_routing_weights;
    }

    auto end_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{number_of_experts_const, first_in_dim, minus_one, last_in_dim},
        0);
    auto end_reshape = std::make_shared<ov::op::v1::Reshape>(down_proj_add, end_shape, true);

    auto mul3 = std::make_shared<ov::op::v1::Multiply>(end_reshape, routing_weights_final);

    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        false);

    ov::ParameterVector params = {input};
    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(reduce_sum)};

    if (additional_node_consumers) {
        results.push_back(std::make_shared<ov::op::v0::Result>(after_tile_reshape));
        results.push_back(std::make_shared<ov::op::v0::Result>(gate_up_add));
        results.push_back(std::make_shared<ov::op::v0::Result>(multiply2));
        results.push_back(std::make_shared<ov::op::v0::Result>(down_proj_add));
    }

    return std::make_shared<ov::Model>(results, params);
}

inline std::shared_ptr<ov::Model> initMoE2GeMMSubgraphRef(bool use_scatter_v12,
                                                          bool use_broadcast_v3,
                                                          bool skip_unsqueeze,
                                                          bool matmul_transpose_b) {
    // Fixed values that don't affect pass behavior
    const auto expert_alpha = 1.625f;
    const auto expert_beta = 7.0f;
    const ov::PartialShape input_shape = {-1, -1, 256};
    const size_t topk = 4;
    const size_t number_of_experts = 8;
    const size_t intermediate_size = 512;
    const ov::element::Type data_precision = ov::element::f32;
    const ov::element::Type weights_precision = ov::element::f32;
    constexpr int64_t fusion_factor = 2;

    auto input = std::make_shared<ov::op::v0::Parameter>(data_precision, input_shape);

    const auto hidden_size = static_cast<size_t>(input_shape[2].get_length());

    auto experts_reshape = std::make_shared<ov::op::v1::Reshape>(
        input,
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto unsqueeze_experts =
        std::make_shared<ov::op::v0::Unsqueeze>(experts_reshape,
                                                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));

    auto router_weights =
        build_matmul_weights(ov::Shape{hidden_size, number_of_experts}, weights_precision, 4, matmul_transpose_b);

    auto reshape_2nd_consumer_router_matmul =
        std::make_shared<ov::op::v0::MatMul>(experts_reshape, router_weights, false, matmul_transpose_b);

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

    // Note: we need to use different seed to avoid the exact weights generation for the MatMuls with the same shape
    size_t seed = 1;
    auto gate_up_weights =
        build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size * fusion_factor},
                             weights_precision,
                             seed++,
                             matmul_transpose_b);

    auto gate_up_bias =
        ov::test::utils::make_constant(data_precision,
                                       ov::Shape{number_of_experts, 1, intermediate_size * fusion_factor});

    auto gate_up_gathered_mm =
        std::make_shared<BatchGatherMatmul>(unsqueeze_experts, gate_up_weights, router_topk_indices, gate_up_bias);

    auto slice1 = std::make_shared<ov::op::v8::Slice>(
        gate_up_gathered_mm,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{1},
                                     std::vector<int64_t>{std::numeric_limits<int64_t>::max()}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}),
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{2}));
    auto clamp = std::make_shared<ov::op::v0::Clamp>(slice1, -expert_beta, expert_beta);
    auto add1 = std::make_shared<ov::op::v1::Add>(clamp, ov::test::utils::make_constant(data_precision, ov::Shape{1}));

    auto slice2 = std::make_shared<ov::op::v8::Slice>(
        gate_up_gathered_mm,
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

    auto down_proj_weights = build_matmul_weights(ov::Shape{number_of_experts, intermediate_size, hidden_size},
                                                  weights_precision,
                                                  seed++,
                                                  matmul_transpose_b);

    auto down_proj_bias = ov::test::utils::make_constant(data_precision, ov::Shape{number_of_experts, 1, hidden_size});

    auto down_gathered_mm =
        std::make_shared<BatchGatherMatmul>(multiply2, down_proj_weights, router_topk_indices, down_proj_bias);

    auto router_transpose = std::make_shared<ov::op::v1::Transpose>(
        slice3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));

    auto router_unsqueeze =
        std::make_shared<ov::op::v0::Unsqueeze>(router_transpose,
                                                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1}));

    auto mul3 = std::make_shared<ov::op::v1::Multiply>(down_gathered_mm, router_unsqueeze);

    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        false);

    const auto number_of_experts_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(number_of_experts)});
    auto first_in_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(input, {0});
    auto last_in_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(input, {2});
    auto minus_one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto end_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{number_of_experts_const, first_in_dim, minus_one, last_in_dim},
        0);

    auto slice_shape =
        std::make_shared<ov::op::v8::Slice>(end_shape,
                                            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4}),
                                            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}));

    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(reduce_sum, slice_shape, true);

    ov::ParameterVector params = {input};
    return std::make_shared<ov::Model>(ov::OutputVector{final_reshape}, params);
}

inline std::shared_ptr<ov::Model> initMoE3GeMMSubgraph(bool use_scatter_v12,
                                                       bool use_broadcast_v3,
                                                       bool skip_unsqueeze,
                                                       bool matmul_transpose_b,
                                                       bool additional_node_consumers) {
    // Fixed values that don't affect pass behavior
    const ov::PartialShape input_shape = {-1, -1, 256};
    const size_t topk = 4;
    const size_t number_of_experts = 8;
    const size_t intermediate_size = 512;
    const ov::element::Type data_precision = ov::element::f32;
    const ov::element::Type weights_precision = ov::element::f32;

    auto input = std::make_shared<ov::op::v0::Parameter>(data_precision, input_shape);
    const auto hidden_size = static_cast<size_t>(input_shape[2].get_length());

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
    auto gate_weights = build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size},
                                             weights_precision,
                                             seed++,
                                             matmul_transpose_b);

    auto gate_matmul =
        std::make_shared<ov::op::v0::MatMul>(after_tile_reshape, gate_weights, false, matmul_transpose_b);

    gate_matmul->set_friendly_name("GateMatMul");

    auto swish = std::make_shared<ov::op::v4::Swish>(gate_matmul);
    auto up_weights = build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size},
                                           weights_precision,
                                           seed++,
                                           matmul_transpose_b);

    auto up_matmul = std::make_shared<ov::op::v0::MatMul>(after_tile_reshape, up_weights, false, matmul_transpose_b);

    up_matmul->set_friendly_name("UpMatMul");

    auto swiglu = std::make_shared<ov::op::v1::Multiply>(swish, up_matmul);

    auto down_weights = build_matmul_weights(ov::Shape{number_of_experts, intermediate_size, hidden_size},
                                             weights_precision,
                                             seed++,
                                             matmul_transpose_b);

    auto down_matmul = std::make_shared<ov::op::v0::MatMul>(swiglu, down_weights, false, matmul_transpose_b);

    down_matmul->set_friendly_name("DownMatMul");

    auto router_weights =
        build_matmul_weights(ov::Shape{hidden_size, number_of_experts}, weights_precision, seed++, matmul_transpose_b);

    auto reshape_2nd_consumer_router_matmul =
        std::make_shared<ov::op::v0::MatMul>(experts_reshape, router_weights, false, matmul_transpose_b);

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
        true);
    auto router_topk_values_normalization =
        std::make_shared<ov::op::v1::Divide>(router_topk_values_and_indices->output(0), router_topk_values_reduce);
    auto router_topk_indices = router_topk_values_and_indices->output(1);

    const auto number_of_experts_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(number_of_experts)});

    auto zero_const =
        ov::op::v0::Constant::create(router_topk_values_normalization->get_output_element_type(0), ov::Shape{1}, {0});
    auto first_topk_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(router_topk_indices, {0});
    auto bcast_target_shape =
        std::make_shared<ov::op::v0::Concat>(ov::OutputVector{first_topk_dim, number_of_experts_const}, 0);

    std::shared_ptr<ov::Node> broadcast_zero;
    if (use_broadcast_v3) {
        broadcast_zero = std::make_shared<ov::op::v3::Broadcast>(zero_const, bcast_target_shape);
    } else {
        broadcast_zero = std::make_shared<ov::op::v1::Broadcast>(zero_const, bcast_target_shape);
    }

    std::shared_ptr<ov::Node> scatter_elements_update;
    if (use_scatter_v12) {
        scatter_elements_update = std::make_shared<ov::op::v12::ScatterElementsUpdate>(
            broadcast_zero,
            router_topk_indices,
            router_topk_values_normalization,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
    } else {
        scatter_elements_update = std::make_shared<ov::op::v3::ScatterElementsUpdate>(
            broadcast_zero,
            router_topk_indices,
            router_topk_values_normalization,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}));
    }

    auto router_transpose = std::make_shared<ov::op::v1::Transpose>(
        scatter_elements_update,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));

    auto minus_one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});
    std::shared_ptr<ov::op::v0::Concat> router_shape;
    if (skip_unsqueeze) {
        auto one_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1});
        router_shape = std::make_shared<ov::op::v0::Concat>(
            ov::OutputVector{number_of_experts_const, first_topk_dim, minus_one, one_const},
            0);
    } else {
        router_shape =
            std::make_shared<ov::op::v0::Concat>(ov::OutputVector{number_of_experts_const, first_topk_dim, minus_one},
                                                 0);
    }

    auto router_reshape = std::make_shared<ov::op::v1::Reshape>(router_transpose, router_shape, true);

    std::shared_ptr<ov::Node> routing_weights_final;
    if (skip_unsqueeze) {
        routing_weights_final = router_reshape;
    } else {
        auto unsqueeze_routing_weights = std::make_shared<ov::op::v0::Unsqueeze>(
            router_reshape,
            ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{3}));
        routing_weights_final = unsqueeze_routing_weights;
    }

    auto last_in_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(input, {2});
    auto end_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{number_of_experts_const, first_topk_dim, minus_one, last_in_dim},
        0);
    auto end_reshape = std::make_shared<ov::op::v1::Reshape>(down_matmul, end_shape, true);

    auto mul3 = std::make_shared<ov::op::v1::Multiply>(end_reshape, routing_weights_final);

    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        false);

    auto final_reshape_const = ov::op::v0::Constant::create(ov::element::i64,
                                                            ov::Shape{2},
                                                            std::vector<int64_t>{0, static_cast<int64_t>(hidden_size)});
    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(reduce_sum, final_reshape_const, true);

    ov::ParameterVector params = {input};
    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(final_reshape)};

    if (additional_node_consumers) {
        results.push_back(std::make_shared<ov::op::v0::Result>(after_tile_reshape));
        results.push_back(std::make_shared<ov::op::v0::Result>(gate_matmul));
        results.push_back(std::make_shared<ov::op::v0::Result>(swish));
        results.push_back(std::make_shared<ov::op::v0::Result>(up_matmul));
        results.push_back(std::make_shared<ov::op::v0::Result>(swiglu));
        results.push_back(std::make_shared<ov::op::v0::Result>(down_matmul));
    }

    return std::make_shared<ov::Model>(results, params);
}

inline std::shared_ptr<ov::Model> initMoE3GeMMSubgraphRef(bool use_scatter_v12,
                                                          bool use_broadcast_v3,
                                                          bool skip_unsqueeze,
                                                          bool matmul_transpose_b) {
    // Fixed values that don't affect pass behavior
    const ov::PartialShape input_shape = {-1, -1, 256};
    const size_t topk = 4;
    const size_t number_of_experts = 8;
    const size_t intermediate_size = 512;
    const ov::element::Type data_precision = ov::element::f32;
    const ov::element::Type weights_precision = ov::element::f32;

    auto input = std::make_shared<ov::op::v0::Parameter>(data_precision, input_shape);

    const auto hidden_size = static_cast<size_t>(input_shape[2].get_length());

    auto experts_reshape = std::make_shared<ov::op::v1::Reshape>(
        input,
        ov::op::v0::Constant::create(ov::element::i64,
                                     ov::Shape{2},
                                     std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto unsqueeze_experts =
        std::make_shared<ov::op::v0::Unsqueeze>(experts_reshape,
                                                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {0}));

    auto router_weights =
        build_matmul_weights(ov::Shape{hidden_size, number_of_experts}, weights_precision, 4, matmul_transpose_b);

    auto reshape_2nd_consumer_router_matmul =
        std::make_shared<ov::op::v0::MatMul>(experts_reshape, router_weights, false, matmul_transpose_b);

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
        true);
    auto router_topk_values_normalization =
        std::make_shared<ov::op::v1::Divide>(router_topk_values_and_indices->output(0), router_topk_values_reduce);
    auto router_topk_indices = router_topk_values_and_indices->output(1);

    // Note: we need to use different seed to avoid the exact weights generation for the MatMuls with the same shape
    size_t seed = 1;
    auto gate_weights = build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size},
                                             weights_precision,
                                             seed++,
                                             matmul_transpose_b);

    auto gate_gathered_mm = std::make_shared<BatchGatherMatmul>(unsqueeze_experts, gate_weights, router_topk_indices);

    auto swish = std::make_shared<ov::op::v4::Swish>(gate_gathered_mm);

    auto up_weights = build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size},
                                           weights_precision,
                                           seed++,
                                           matmul_transpose_b);

    auto up_gathered_mm = std::make_shared<BatchGatherMatmul>(unsqueeze_experts, up_weights, router_topk_indices);

    auto swiglu = std::make_shared<ov::op::v1::Multiply>(swish, up_gathered_mm);

    auto down_weights = build_matmul_weights(ov::Shape{number_of_experts, intermediate_size, hidden_size},
                                             weights_precision,
                                             seed++,
                                             matmul_transpose_b);

    auto down_gathered_mm = std::make_shared<BatchGatherMatmul>(swiglu, down_weights, router_topk_indices);

    auto router_transpose = std::make_shared<ov::op::v1::Transpose>(
        router_topk_values_normalization,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0}));

    auto router_unsqueeze =
        std::make_shared<ov::op::v0::Unsqueeze>(router_transpose,
                                                ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1}));

    auto mul3 = std::make_shared<ov::op::v1::Multiply>(down_gathered_mm, router_unsqueeze);

    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        false);

    const auto number_of_experts_const =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {static_cast<int64_t>(number_of_experts)});
    auto first_topk_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(router_topk_indices, {0});
    auto last_in_dim = ov::op::util::node_to_get_shape_value_of_indices_from_shape_source(input, {2});
    auto minus_one = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1});

    auto end_shape = std::make_shared<ov::op::v0::Concat>(
        ov::OutputVector{number_of_experts_const, first_topk_dim, minus_one, last_in_dim},
        0);

    auto slice_shape =
        std::make_shared<ov::op::v8::Slice>(end_shape,
                                            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {4}),
                                            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {1}),
                                            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{1}, {0}));

    auto final_reshape = std::make_shared<ov::op::v1::Reshape>(reduce_sum, slice_shape, true);

    auto final_reshape_const = ov::op::v0::Constant::create(ov::element::i64,
                                                            ov::Shape{2},
                                                            std::vector<int64_t>{0, static_cast<int64_t>(hidden_size)});
    auto original_final_reshape = std::make_shared<ov::op::v1::Reshape>(final_reshape, final_reshape_const, true);

    ov::ParameterVector params = {input};
    return std::make_shared<ov::Model>(ov::OutputVector{original_final_reshape}, params);
}

class MoEMatMulsFusionTest : public TransformationTestsF, public WithParamInterface<MoEMatMulsFusionParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<MoEMatMulsFusionParams>& obj) {
        const auto& [moe_type,
                     use_scatter_v12,
                     use_broadcast_v3,
                     skip_unsqueeze,
                     matmul_transpose_b,
                     additional_node_consumers,
                     should_be_applied] = obj.param;
        std::ostringstream result;
        result << "MoEType_" << moe_type << "_ScatterV" << (use_scatter_v12 ? "12" : "3") << "_BroadcastV"
               << (use_broadcast_v3 ? "3" : "1") << "_SkipUnsqueeze_" << (skip_unsqueeze ? "true" : "false")
               << "_MatMulTransposeB_" << (matmul_transpose_b ? "true" : "false") << "_AdditionalConsumers_"
               << (additional_node_consumers ? "true" : "false") << "_shouldBeApplied_"
               << (should_be_applied ? "true" : "false");
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [moe_type,
                     use_scatter_v12,
                     use_broadcast_v3,
                     skip_unsqueeze,
                     matmul_transpose_b,
                     additional_node_consumers,
                     should_be_applied] = this->GetParam();

        switch (moe_type) {
        case MoEType::MoE2GeMM:
            model = initMoE2GeMMSubgraph(use_scatter_v12,
                                         use_broadcast_v3,
                                         skip_unsqueeze,
                                         matmul_transpose_b,
                                         additional_node_consumers);
            break;
        case MoEType::MoE3GeMM:
            model = initMoE3GeMMSubgraph(use_scatter_v12,
                                         use_broadcast_v3,
                                         skip_unsqueeze,
                                         matmul_transpose_b,
                                         additional_node_consumers);
            break;
        default:
            OPENVINO_THROW("Unexpected MoEType value");
        }

        manager.register_pass<MoEMatMulsFusion>();

        if (should_be_applied) {
            switch (moe_type) {
            case MoEType::MoE2GeMM:
                model_ref =
                    initMoE2GeMMSubgraphRef(use_scatter_v12, use_broadcast_v3, skip_unsqueeze, matmul_transpose_b);
                break;
            case MoEType::MoE3GeMM:
                model_ref =
                    initMoE3GeMMSubgraphRef(use_scatter_v12, use_broadcast_v3, skip_unsqueeze, matmul_transpose_b);
                break;
            default:
                OPENVINO_THROW("Unexpected MoEType value");
            }
        }
    }
};

TEST_P(MoEMatMulsFusionTest, CompareFunctions) {}

const std::vector<MoEType> moe_types = {MoEType::MoE2GeMM, MoEType::MoE3GeMM};
const std::vector<bool> scatter_versions = {false, true};    // false = v3, true = v12
const std::vector<bool> broadcast_versions = {false, true};  // false = v1, true = v3
const std::vector<bool> skip_unsqueeze_versions = {false, true};

INSTANTIATE_TEST_SUITE_P(MoEMatMulsFusionTest_positive_cases,
                         MoEMatMulsFusionTest,
                         ::testing::Combine(::testing::ValuesIn(moe_types),
                                            ::testing::ValuesIn(scatter_versions),
                                            ::testing::ValuesIn(broadcast_versions),
                                            ::testing::ValuesIn(skip_unsqueeze_versions),
                                            ::testing::Values(true),
                                            ::testing::Values(false),
                                            ::testing::Values(true)),
                         MoEMatMulsFusionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MoEMatMulsFusionTest_negative_cases_no_transpose,
                         MoEMatMulsFusionTest,
                         ::testing::Combine(::testing::ValuesIn(moe_types),
                                            ::testing::ValuesIn(scatter_versions),
                                            ::testing::ValuesIn(broadcast_versions),
                                            ::testing::ValuesIn(skip_unsqueeze_versions),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false)),
                         MoEMatMulsFusionTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(MoEMatMulsFusionTest_negative_cases_additional_consumers,
                         MoEMatMulsFusionTest,
                         ::testing::Combine(::testing::ValuesIn(moe_types),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(false),
                                            ::testing::Values(true),
                                            ::testing::Values(true),
                                            ::testing::Values(false)),
                         MoEMatMulsFusionTest::getTestCaseName);
}  // namespace