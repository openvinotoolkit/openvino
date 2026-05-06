// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <sstream>
#include <string>

#include "common_test_utils/node_builders/constant.hpp"
#include "common_test_utils/ov_test_utils.hpp"
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
#include "ov_ops/gather_matmul.hpp"
#include "transformations/utils/utils.hpp"

using namespace testing;
using namespace ov::op::internal;

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

enum class AdditionalConsumersMode { NO, MATMULS, AFTER_MATMULS, REDUCE };

inline std::ostream& operator<<(std::ostream& os, const AdditionalConsumersMode& mode) {
    switch (mode) {
    case AdditionalConsumersMode::NO:
        return os << "NO";
    case AdditionalConsumersMode::MATMULS:
        return os << "MATMULS";
    case AdditionalConsumersMode::AFTER_MATMULS:
        return os << "AFTER_MATMULS";
    case AdditionalConsumersMode::REDUCE:
        return os << "REDUCE";
    default:
        OPENVINO_THROW("Unsupported AdditionalConsumersMode");
    }
}

using ConvertTiledMoeBlockToGatherMatmulsParams = std::tuple<MoEType,                  // moe_type
                                                             bool,                     // matmul_transpose_b
                                                             AdditionalConsumersMode,  // additional_consumers_mode
                                                             bool>;  // transformation_should_be_applied

inline std::shared_ptr<ov::Node> build_matmul_weights(const ov::Shape& weights_shape,
                                                      const ov::element::Type& weights_precision,
                                                      int seed,
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

inline std::shared_ptr<ov::Model> initMoE2GeMMSubgraph(bool matmul_transpose_b,
                                                       AdditionalConsumersMode additional_consumers_mode) {
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
    int seed = 1;
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

    // Router (new form): Transpose[1,0,2] -> Gather(axis=1, batch_dims=1, topk)
    auto eb_to_be_perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2});
    auto router_transpose = std::make_shared<ov::op::v1::Transpose>(down_proj_add, eb_to_be_perm);
    auto gather_axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto router_gather = std::make_shared<ov::op::v8::Gather>(router_transpose,
                                                              router_topk_indices,
                                                              gather_axis_const,
                                                              /*batch_dims=*/1);

    auto routing_weights_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
        router_topk_values_softmax,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}));

    auto mul3 = std::make_shared<ov::op::v1::Multiply>(router_gather, routing_weights_unsqueeze);

    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
        false);

    ov::ParameterVector params = {input};
    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(reduce_sum)};

    if (additional_consumers_mode == AdditionalConsumersMode::MATMULS) {
        results.push_back(std::make_shared<ov::op::v0::Result>(gate_up_matmul));
        results.push_back(std::make_shared<ov::op::v0::Result>(down_proj_matmul));
    } else if (additional_consumers_mode == AdditionalConsumersMode::AFTER_MATMULS) {
        results.push_back(std::make_shared<ov::op::v0::Result>(add1));
    } else if (additional_consumers_mode == AdditionalConsumersMode::REDUCE) {
        results.push_back(std::make_shared<ov::op::v0::Result>(reduce_sum));
    }

    return std::make_shared<ov::Model>(results, params);
}

inline std::shared_ptr<ov::Model> initMoE2GeMMSubgraphRef(bool matmul_transpose_b) {
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

    // Note: we need to use different seed to avoid the exact weights generation for the MatMuls with the same shape
    int seed = 1;
    auto gate_up_weights =
        build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size * fusion_factor},
                             weights_precision,
                             seed++,
                             matmul_transpose_b);

    auto gate_up_bias =
        ov::test::utils::make_constant(data_precision,
                                       ov::Shape{number_of_experts, 1, intermediate_size * fusion_factor});

    auto gate_up_gathered_mm =
        std::make_shared<GatherMatmul>(unsqueeze_experts, gate_up_weights, router_topk_indices, gate_up_bias);

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
        std::make_shared<GatherMatmul>(multiply2, down_proj_weights, router_topk_indices, down_proj_bias);

    // Routing weights: [B,K] -> Transpose([1,0]) -> [K,B] -> Unsqueeze(-1) -> [K,B,1]
    auto routing_transpose_perm =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});
    auto routing_transposed =
        std::make_shared<ov::op::v1::Transpose>(router_topk_values_softmax, routing_transpose_perm);
    auto routing_weights_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
        routing_transposed,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, std::vector<int32_t>{-1}));

    // Multiply [K,B,H_out] * [K,B,1] -> ReduceSum axis=0 -> [B,H_out]
    auto mul3 = std::make_shared<ov::op::v1::Multiply>(down_gathered_mm, routing_weights_unsqueeze);

    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        false);

    ov::ParameterVector params = {input};
    return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum}, params);
}

inline std::shared_ptr<ov::Model> initMoE3GeMMSubgraph(bool matmul_transpose_b,
                                                       AdditionalConsumersMode additional_consumers_mode) {
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
    int seed = 1;
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

    // Router (new form): Transpose[1,0,2] -> Gather(axis=1, batch_dims=1, topk)
    auto eb_to_be_perm = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{3}, std::vector<int64_t>{1, 0, 2});
    auto router_transpose = std::make_shared<ov::op::v1::Transpose>(down_matmul, eb_to_be_perm);
    auto gather_axis_const = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, std::vector<int64_t>{1});
    auto router_gather = std::make_shared<ov::op::v8::Gather>(router_transpose,
                                                              router_topk_indices,
                                                              gather_axis_const,
                                                              /*batch_dims=*/1);

    auto routing_weights_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
        router_topk_values_normalization,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{-1}));

    auto mul3 = std::make_shared<ov::op::v1::Multiply>(router_gather, routing_weights_unsqueeze);

    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{1}),
        false);

    ov::ParameterVector params = {input};
    ov::ResultVector results = {std::make_shared<ov::op::v0::Result>(reduce_sum)};

    if (additional_consumers_mode == AdditionalConsumersMode::MATMULS) {
        results.push_back(std::make_shared<ov::op::v0::Result>(gate_matmul));
        results.push_back(std::make_shared<ov::op::v0::Result>(up_matmul));
        results.push_back(std::make_shared<ov::op::v0::Result>(down_matmul));
    } else if (additional_consumers_mode == AdditionalConsumersMode::AFTER_MATMULS) {
        results.push_back(std::make_shared<ov::op::v0::Result>(swish));
        results.push_back(std::make_shared<ov::op::v0::Result>(swiglu));
    } else if (additional_consumers_mode == AdditionalConsumersMode::REDUCE) {
        results.push_back(std::make_shared<ov::op::v0::Result>(reduce_sum));
    }

    return std::make_shared<ov::Model>(results, params);
}

inline std::shared_ptr<ov::Model> initMoE3GeMMSubgraphRef(bool matmul_transpose_b) {
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
    int seed = 1;
    auto gate_weights = build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size},
                                             weights_precision,
                                             seed++,
                                             matmul_transpose_b);

    auto gate_gathered_mm = std::make_shared<GatherMatmul>(unsqueeze_experts, gate_weights, router_topk_indices);

    auto swish = std::make_shared<ov::op::v4::Swish>(gate_gathered_mm);

    auto up_weights = build_matmul_weights(ov::Shape{number_of_experts, hidden_size, intermediate_size},
                                           weights_precision,
                                           seed++,
                                           matmul_transpose_b);

    auto up_gathered_mm = std::make_shared<GatherMatmul>(unsqueeze_experts, up_weights, router_topk_indices);

    auto swiglu = std::make_shared<ov::op::v1::Multiply>(swish, up_gathered_mm);

    auto down_weights = build_matmul_weights(ov::Shape{number_of_experts, intermediate_size, hidden_size},
                                             weights_precision,
                                             seed++,
                                             matmul_transpose_b);

    auto down_gathered_mm = std::make_shared<GatherMatmul>(swiglu, down_weights, router_topk_indices);

    // Routing weights: [B,K] -> Transpose([1,0]) -> [K,B] -> Unsqueeze(-1) -> [K,B,1]
    auto routing_transpose_perm =
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, std::vector<int64_t>{1, 0});
    auto routing_transposed =
        std::make_shared<ov::op::v1::Transpose>(router_topk_values_normalization, routing_transpose_perm);
    auto routing_weights_unsqueeze = std::make_shared<ov::op::v0::Unsqueeze>(
        routing_transposed,
        ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, std::vector<int32_t>{-1}));

    // Multiply [K,B,H_out] * [K,B,1] -> ReduceSum axis=0 -> [B,H_out]
    auto mul3 = std::make_shared<ov::op::v1::Multiply>(down_gathered_mm, routing_weights_unsqueeze);

    auto reduce_sum = std::make_shared<ov::op::v1::ReduceSum>(
        mul3,
        ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, std::vector<int64_t>{0}),
        false);

    ov::ParameterVector params = {input};
    return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum}, params);
}

class ConvertTiledMoeBlockToGatherMatmulsTest : public TransformationTestsF,
                                                public WithParamInterface<ConvertTiledMoeBlockToGatherMatmulsParams> {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<ConvertTiledMoeBlockToGatherMatmulsParams>& obj) {
        const auto& [moe_type, matmul_transpose_b, additional_consumers_mode, should_be_applied] = obj.param;
        std::ostringstream result;
        result << "MoEType_" << moe_type << "_MatMulTransposeB_" << (matmul_transpose_b ? "true" : "false")
               << "_AdditionalConsumers_" << additional_consumers_mode << "_shouldBeApplied_"
               << (should_be_applied ? "true" : "false");
        return result.str();
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        const auto& [moe_type, matmul_transpose_b, additional_consumers_mode, should_be_applied] = this->GetParam();

        switch (moe_type) {
        case MoEType::MoE2GeMM:
            model = initMoE2GeMMSubgraph(matmul_transpose_b, additional_consumers_mode);
            break;
        case MoEType::MoE3GeMM:
            model = initMoE3GeMMSubgraph(matmul_transpose_b, additional_consumers_mode);
            break;
        default:
            OPENVINO_THROW("Unexpected MoEType value");
        }

        manager.register_pass<ov::pass::ConvertTiledMoeBlockToGatherMatmuls>();

        if (should_be_applied) {
            switch (moe_type) {
            case MoEType::MoE2GeMM:
                model_ref = initMoE2GeMMSubgraphRef(matmul_transpose_b);
                break;
            case MoEType::MoE3GeMM:
                model_ref = initMoE3GeMMSubgraphRef(matmul_transpose_b);
                break;
            default:
                OPENVINO_THROW("Unexpected MoEType value");
            }
        }
    }
};

TEST_P(ConvertTiledMoeBlockToGatherMatmulsTest, CompareFunctions) {}

const std::vector<MoEType> moe_types = {MoEType::MoE2GeMM, MoEType::MoE3GeMM};

INSTANTIATE_TEST_SUITE_P(ConvertTiledMoeBlockToGatherMatmulsTest_positive_cases,
                         ConvertTiledMoeBlockToGatherMatmulsTest,
                         ::testing::Combine(::testing::ValuesIn(moe_types),
                                            ::testing::Values(true),
                                            ::testing::Values(AdditionalConsumersMode::NO),
                                            ::testing::Values(true)),
                         ConvertTiledMoeBlockToGatherMatmulsTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ConvertTiledMoeBlockToGatherMatmulsTest_negative_cases_no_transpose,
                         ConvertTiledMoeBlockToGatherMatmulsTest,
                         ::testing::Combine(::testing::ValuesIn(moe_types),
                                            ::testing::Values(false),
                                            ::testing::Values(AdditionalConsumersMode::NO),
                                            ::testing::Values(false)),
                         ConvertTiledMoeBlockToGatherMatmulsTest::getTestCaseName);

INSTANTIATE_TEST_SUITE_P(ConvertTiledMoeBlockToGatherMatmulsTest_negative_cases_additional_consumers,
                         ConvertTiledMoeBlockToGatherMatmulsTest,
                         ::testing::Combine(::testing::ValuesIn(moe_types),
                                            ::testing::Values(true),
                                            ::testing::Values(AdditionalConsumersMode::MATMULS,
                                                              AdditionalConsumersMode::AFTER_MATMULS,
                                                              AdditionalConsumersMode::REDUCE),
                                            ::testing::Values(false)),
                         ConvertTiledMoeBlockToGatherMatmulsTest::getTestCaseName);
}  // namespace
