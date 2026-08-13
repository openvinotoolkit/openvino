// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <algorithm>
#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/concat.hpp>
#include <openvino/op/constant.hpp>
#include <openvino/op/gelu.hpp>
#include <openvino/op/matmul.hpp>
#include <openvino/op/minimum.hpp>
#include <openvino/op/moe.hpp>
#include <openvino/op/multiply.hpp>
#include <openvino/op/parameter.hpp>
#include <openvino/op/reduce_sum.hpp>
#include <openvino/op/reshape.hpp>
#include <openvino/op/slice.hpp>
#include <openvino/op/swish.hpp>
#include <openvino/op/tile.hpp>
#include <openvino/op/topk.hpp>
#include <openvino/pass/manager.hpp>
#include <openvino/pass/serialize.hpp>
#include <openvino/pass/visualize_tree.hpp>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "ov_ops/gather_matmul.hpp"
#include "ov_ops/gather_matmul_compressed.hpp"
#include "ov_ops/moe_compressed.hpp"
#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"
#include "transformations/common_optimizations/moe_op_fusion.hpp"

using GatherMatmul = ov::op::internal::GatherMatmul;
using GatherMatmulCompressed = ov::op::internal::GatherMatmulCompressed;
using MOECompressed = ov::op::internal::MOECompressed;

// ============================================================================
// IR model builders (original MOE pattern before any transformation)
// ============================================================================

inline std::shared_ptr<ov::Model> build_2gemm_moe_pattern_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t topk = 2;
    const size_t number_of_experts = 3;
    const size_t fusion_factor = 2;
    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    auto input_shape = PartialShape{batch, in_dim, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, hidden_size}),
        false);

    auto tile = std::make_shared<op::v0::Tile>(
        experts_reshape,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{number_of_experts, 1}));
    auto after_tile_reshape = std::make_shared<op::v1::Reshape>(
        tile,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, hidden_size}),
        false);

    auto gate_up_matmul = std::make_shared<op::v0::MatMul>(
        after_tile_reshape,
        op::v0::Constant::create(element::f32,
                                 Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                 {1.0f}),
        false,
        true);
    auto gate_up_add = std::make_shared<op::v1::Add>(
        gate_up_matmul,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f}));

    auto slice1 = std::make_shared<op::v8::Slice>(
        gate_up_add,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{number_of_experts, batch, intermediate_size * 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto clamp = std::make_shared<op::v0::Clamp>(slice1, -expert_beta, expert_beta);
    auto add1 = std::make_shared<op::v1::Add>(clamp, op::v0::Constant::create(element::f32, Shape{1}, {1.0f}));

    auto slice2 = std::make_shared<op::v8::Slice>(
        gate_up_add,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{number_of_experts, batch, intermediate_size * 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto minimum1 =
        std::make_shared<op::v1::Minimum>(slice2, op::v0::Constant::create(element::f32, Shape{1}, {10.0f}));
    auto swish_beta = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{expert_alpha});
    auto swish = std::make_shared<op::v4::Swish>(minimum1, swish_beta);

    auto multiply2 = std::make_shared<op::v1::Multiply>(add1, swish);

    auto down_proj_matmul = std::make_shared<op::v0::MatMul>(
        multiply2,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f}),
        false,
        true);

    auto down_proj_add = std::make_shared<op::v1::Add>(
        down_proj_matmul,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f}));

    auto end_reshape = std::make_shared<op::v1::Reshape>(
        down_proj_add,
        op::v0::Constant::create(element::i64,
                                 Shape{4},
                                 std::vector<int64_t>{number_of_experts, batch, -1, hidden_size}),
        false);

    // Router subgraph
    auto reshape_2nd_consumer_router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);

    auto router_bias =
        std::make_shared<op::v1::Add>(reshape_2nd_consumer_router_matmul,
                                      op::v0::Constant::create(element::f32, Shape{1, number_of_experts}, {1.0f}));

    auto router_topk_values_and_indices =
        std::make_shared<op::v11::TopK>(router_bias,
                                        op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                        -1,
                                        op::v11::TopK::Mode::MAX,
                                        op::v11::TopK::SortType::SORT_VALUES,
                                        element::i64);

    auto router_topk_values = router_topk_values_and_indices->output(0);
    auto router_topk_indices = router_topk_values_and_indices->output(1);

    auto scatter_elements_update = std::make_shared<op::v12::ScatterElementsUpdate>(
        router_topk_values,
        router_topk_indices,
        op::v0::Constant::create(element::f32, Shape{batch, topk}, {0}),
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        scatter_elements_update,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_reshape = std::make_shared<op::v1::Reshape>(
        router_transpose,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, -1}),
        true);
    auto unsqueeze_routing_weights =
        std::make_shared<op::v0::Unsqueeze>(router_reshape,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    auto mul3 = std::make_shared<op::v1::Multiply>(end_reshape, unsqueeze_routing_weights);

    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(mul3,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_3gemm_moe_pattern_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input_shape = PartialShape{batch, in_dim, hidden_size};
    auto input = std::make_shared<op::v0::Parameter>(element::f32, input_shape);
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, hidden_size}),
        false);

    auto tile = std::make_shared<op::v0::Tile>(
        experts_reshape,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{number_of_experts, 1}));
    auto after_tile_reshape = std::make_shared<op::v1::Reshape>(
        tile,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, hidden_size}),
        false);

    // First GEMM (gate)
    auto gate_matmul = std::make_shared<op::v0::MatMul>(
        after_tile_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f}),
        false,
        true);

    auto swish = std::make_shared<op::v4::Swish>(gate_matmul);

    // Second GEMM (up)
    auto up_matmul = std::make_shared<op::v0::MatMul>(
        after_tile_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f}),
        false,
        true);

    auto swiglu = std::make_shared<op::v1::Multiply>(swish, up_matmul);

    // Third GEMM (down)
    auto down_matmul = std::make_shared<op::v0::MatMul>(
        swiglu,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f}),
        false,
        true);

    auto experts_out_reshape = std::make_shared<op::v1::Reshape>(
        down_matmul,
        op::v0::Constant::create(element::i64,
                                 Shape{4},
                                 std::vector<int64_t>{number_of_experts, batch, -1, hidden_size}),
        false);

    // Router subgraph
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);

    auto router_topk_values_and_indices =
        std::make_shared<op::v11::TopK>(router_matmul,
                                        op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                        -1,
                                        op::v11::TopK::Mode::MAX,
                                        op::v11::TopK::SortType::SORT_VALUES,
                                        element::i64);

    auto router_topk_values = router_topk_values_and_indices->output(0);
    auto router_topk_indices = router_topk_values_and_indices->output(1);

    auto scatter_elements_update = std::make_shared<op::v12::ScatterElementsUpdate>(
        router_topk_values,
        router_topk_indices,
        op::v0::Constant::create(element::f32, Shape{batch, topk}, {0}),
        op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        scatter_elements_update,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_reshape = std::make_shared<op::v1::Reshape>(
        router_transpose,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{number_of_experts, batch, -1}),
        true);
    auto unsqueeze_routing_weights =
        std::make_shared<op::v0::Unsqueeze>(router_reshape,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    auto mul3 = std::make_shared<op::v1::Multiply>(experts_out_reshape, unsqueeze_routing_weights);

    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(mul3,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum}, ov::ParameterVector{input});
}

// ============================================================================
// Post-BGM model builders (3 BGMs + compact routing + ReduceSum + Reshape)
// ============================================================================

inline std::shared_ptr<ov::Model> build_3gemm_bgm_model(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    // Unsqueeze to add experts dimension: [1, batch*seq, hidden]
    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router subgraph to produce topk_indices and chosen_experts
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);    // [batch*seq, topk]
    auto chosen_experts = router_topk->output(0);  // [batch*seq, topk] (values used as routing weights)

    // Gate weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    // Up weights
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    // Down weights
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // 3 BGMs
    auto bgm_gate = std::make_shared<GatherMatmul>(unsqueeze, gate_w, topk_indices);
    std::shared_ptr<ov::Node> gate_act;
    if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_TANH) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::TANH);
    } else if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_ERF) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::ERF);
    } else {
        gate_act = std::make_shared<op::v4::Swish>(bgm_gate);
    }
    auto bgm_up = std::make_shared<GatherMatmul>(unsqueeze, up_w, topk_indices);
    auto swiglu = std::make_shared<op::v1::Multiply>(gate_act, bgm_up);
    auto bgm_down = std::make_shared<GatherMatmul>(swiglu, down_w, topk_indices);

    // Compact routing: chosen_experts → Transpose({1,0}) → Unsqueeze(-1)
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Final: Multiply → ReduceSum → Reshape
    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(
            element::i64,
            Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_3gemm_bgm_to_moe_reference_model(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    // Router subgraph (not fused, remains in the graph)
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto routing = router_topk->output(0);

    // Weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // MOE op with compact routing
    ov::OutputVector moe_inputs = {input, routing, topk_indices, gate_w, up_w, down_w};
    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    config.activation_type = activation_type;
    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);

    return std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_2gemm_bgm_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t topk = 2;
    const size_t number_of_experts = 3;
    const size_t fusion_factor = 2;
    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_bias =
        std::make_shared<op::v1::Add>(router_matmul,
                                      op::v0::Constant::create(element::f32, Shape{1, number_of_experts}, {1.0f}));
    auto router_topk = std::make_shared<op::v11::TopK>(router_bias,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Gate/up weights and bias
    auto gate_up_w = op::v0::Constant::create(element::f32,
                                              Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                              {1.0f});
    auto gate_up_bias =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f});

    // BGM gate_up (4 inputs: data, weight, indices, bias)
    auto bgm_gate_up = std::make_shared<GatherMatmul>(unsqueeze, gate_up_w, topk_indices, gate_up_bias);

    // Activation subgraph (same as in the original 2GEMM pattern)
    auto slice1 = std::make_shared<op::v8::Slice>(
        bgm_gate_up,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(topk),
                                                      static_cast<int64_t>(batch),
                                                      static_cast<int64_t>(intermediate_size * 2)}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto clamp = std::make_shared<op::v0::Clamp>(slice1, -expert_beta, expert_beta);
    auto add1 = std::make_shared<op::v1::Add>(clamp, op::v0::Constant::create(element::f32, Shape{1}, {1.0f}));

    auto slice2 = std::make_shared<op::v8::Slice>(
        bgm_gate_up,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(topk),
                                                      static_cast<int64_t>(batch),
                                                      static_cast<int64_t>(intermediate_size * 2)}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto minimum1 =
        std::make_shared<op::v1::Minimum>(slice2, op::v0::Constant::create(element::f32, Shape{1}, {10.0f}));
    auto swish_beta_const = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{expert_alpha});
    auto swish = std::make_shared<op::v4::Swish>(minimum1, swish_beta_const);
    auto multiply2 = std::make_shared<op::v1::Multiply>(add1, swish);

    // Down proj
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});
    auto down_bias = op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f});
    auto bgm_down = std::make_shared<GatherMatmul>(multiply2, down_w, topk_indices, down_bias);

    // Compact routing
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);
    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(
            element::i64,
            Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_2gemm_bgm_to_moe_reference_model() {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t topk = 2;
    const size_t number_of_experts = 3;
    const size_t fusion_factor = 2;
    const auto expert_alpha = 1.702f;
    const auto expert_beta = 7.0f;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    // Router (stays in graph)
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_bias =
        std::make_shared<op::v1::Add>(router_matmul,
                                      op::v0::Constant::create(element::f32, Shape{1, number_of_experts}, {1.0f}));
    auto router_topk = std::make_shared<op::v11::TopK>(router_bias,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Convert2GatherMatmulMoeBlockToMoeOp bypasses Transpose+Unsqueeze (tokens-major).
    auto routing = chosen_experts;

    // Weights
    auto gate_up_w = op::v0::Constant::create(element::f32,
                                              Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                              {1.0f});
    auto gate_up_bias =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});
    auto down_bias = op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f});

    ov::OutputVector moe_inputs = {input, routing, topk_indices, gate_up_w, gate_up_bias, down_w, down_bias};

    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;
    config.expert_alpha = expert_alpha;
    config.expert_beta = expert_beta;

    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
    return std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{input});
}

// ============================================================================
// Post-BGM model builders (3 BGMs + compact routing) — Multiply instead of Reshape
// before Unsqueeze (gemma4 pattern: layernorm Multiply feeds expert path directly)
// ============================================================================

inline std::shared_ptr<ov::Model> build_3gemm_bgm_model_multiply_input(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    // Simulate layernorm output: Multiply(input, scale) — no Reshape
    auto norm_scale = op::v0::Constant::create(element::f32, Shape{1, 1, hidden_size}, {1.0f});
    auto layernorm_mul = std::make_shared<op::v1::Multiply>(input, norm_scale);

    // Reshape to [batch*seq, hidden] for router path
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        layernorm_mul,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    // Unsqueeze feeds from Multiply (via Reshape) — but the pattern's optional<Reshape>
    // means hidden_states_m captures the Multiply output (layernorm_mul).
    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    // Router subgraph
    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    // Weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // 3 BGMs
    auto bgm_gate = std::make_shared<GatherMatmul>(unsqueeze, gate_w, topk_indices);
    std::shared_ptr<ov::Node> gate_act;
    if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_TANH) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::TANH);
    } else if (activation_type == ov::op::internal::MOE::Activation_type::GEGLU_ERF) {
        gate_act = std::make_shared<op::v7::Gelu>(bgm_gate, ov::op::GeluApproximationMode::ERF);
    } else {
        gate_act = std::make_shared<op::v4::Swish>(bgm_gate);
    }
    auto bgm_up = std::make_shared<GatherMatmul>(unsqueeze, up_w, topk_indices);
    auto swiglu = std::make_shared<op::v1::Multiply>(gate_act, bgm_up);
    auto bgm_down = std::make_shared<GatherMatmul>(swiglu, down_w, topk_indices);

    // Compact routing
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Final: Multiply → ReduceSum → Reshape
    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(
            element::i64,
            Shape{3},
            std::vector<int64_t>{static_cast<int64_t>(batch), -1, static_cast<int64_t>(hidden_size)}),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_3gemm_bgm_to_moe_reference_model_multiply_input(
    ov::op::internal::MOE::Activation_type activation_type = ov::op::internal::MOE::Activation_type::SWIGLU) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t number_of_experts = 3;
    const size_t topk = 2;

    auto input = std::make_shared<op::v0::Parameter>(element::f32, PartialShape{batch, in_dim, hidden_size});

    // Layernorm Multiply — its OUTPUT is the correct hidden_states for the MOE op
    auto norm_scale = op::v0::Constant::create(element::f32, Shape{1, 1, hidden_size}, {1.0f});
    auto layernorm_mul = std::make_shared<op::v1::Multiply>(input, norm_scale);

    // Router subgraph (not fused, remains in the graph)
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        layernorm_mul,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(hidden_size)}),
        false);

    auto router_matmul = std::make_shared<op::v0::MatMul>(
        experts_reshape,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size}, {1.0f}),
        false,
        true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                       op::v0::Constant::create(element::i64, Shape{}, {topk}),
                                                       -1,
                                                       op::v11::TopK::Mode::MAX,
                                                       op::v11::TopK::SortType::SORT_VALUES,
                                                       element::i64);
    auto topk_indices = router_topk->output(1);
    auto routing = router_topk->output(0);

    // Weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // MOE op — hidden_states input is the layernorm Multiply output (NOT the Parameter)
    ov::OutputVector moe_inputs = {layernorm_mul, routing, topk_indices, gate_w, up_w, down_w};
    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
    config.activation_type = activation_type;
    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);

    return std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{input});
}

// ============================================================================
// Tests for BGM→MOE passes (Convert3GatherMatmulMoeBlockToMoeOp, Convert2GatherMatmulMoeBlockToMoeOp)
// ============================================================================

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_basic) {
    model = build_3gemm_bgm_model();
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model();
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_gelu_tanh) {
    using AT = ov::op::internal::MOE::Activation_type;
    model = build_3gemm_bgm_model(AT::GEGLU_TANH);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::GEGLU_TANH);
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_gelu_erf) {
    using AT = ov::op::internal::MOE::Activation_type;
    model = build_3gemm_bgm_model(AT::GEGLU_ERF);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model(AT::GEGLU_ERF);
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_multiply_input) {
    model = build_3gemm_bgm_model_multiply_input();
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model_multiply_input();
}

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_multiply_input_gelu_tanh) {
    using AT = ov::op::internal::MOE::Activation_type;
    model = build_3gemm_bgm_model_multiply_input(AT::GEGLU_TANH);
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_3gemm_bgm_to_moe_reference_model_multiply_input(AT::GEGLU_TANH);
}

TEST_F(TransformationTestsF, Convert2GatherMatmulMoeBlockToMoeOp_basic) {
    model = build_2gemm_bgm_model();
    manager.register_pass<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_2gemm_bgm_to_moe_reference_model();
}

// ============================================================================
// Compressed (GatherMatmulCompressed) variants.
//
// Every builder above produces PLAIN GatherMatmul with f32 weights, so `is_compressed` is false
// and the entire compressed half of the callback -- the weight-dtype admission gate, the scalar-zp
// normalization, the group_size unification and the per-channel scale/zp broadcast -- is
// unreachable from the tests in this file. The models below exist to reach it.
// ============================================================================

namespace {

// The compressed branch only inspects weight dtypes and scale/zp shapes, so the smallest shapes
// that still express the interesting cases are enough. gate/up carry 4 scale groups over K=16 and
// down carries 2 over K=8, i.e. both sides must unify to the same group_size of 4 -- a single G
// everywhere would not exercise that.
constexpr size_t kE = 2;
constexpr size_t kHidden = 16;
constexpr size_t kInter = 8;
constexpr size_t kTopk = 2;
constexpr size_t kGroupSize = 4;

// An absent optional input. GatherMatmul itself spells a missing bias this way; note it has to be
// the 2-argument Constant ctor, since Constant::create cannot produce a dynamic element type.
std::shared_ptr<ov::Node> absent_input() {
    return std::make_shared<ov::op::v0::Constant>(ov::element::dynamic, ov::Shape{0});
}

// Quantized values are never dequantized by these tests, but they must still be distinct per
// element so that a wrong broadcast/expansion is visible rather than masked by a uniform fill.
// Constant::create range-checks every element, and u2 holds only 0..3, so the pattern is clamped
// to whatever the requested type can actually represent.
std::shared_ptr<ov::Node> int_const(ov::element::Type dt, const ov::Shape& shape, int32_t base = 0) {
    const int32_t hi = dt.bitwidth() < 8
                           ? static_cast<int32_t>((1u << (dt.bitwidth() - (dt.is_signed() ? 1 : 0))) - 1)
                           : 127;
    const size_t n = ov::shape_size(shape);
    std::vector<int32_t> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = std::min(base + static_cast<int32_t>(i % 4), hi);
    }
    return ov::op::v0::Constant::create(dt, shape, v);
}

std::shared_ptr<ov::Node> scale_const(const ov::Shape& shape) {
    const size_t n = ov::shape_size(shape);
    std::vector<float> v(n);
    for (size_t i = 0; i < n; ++i) {
        v[i] = 0.5f + 0.25f * static_cast<float>(i);
    }
    return ov::op::v0::Constant::create(ov::element::f16, shape, v);
}

struct CompressedMoeSpec {
    ov::element::Type gate_dt = ov::element::u4;
    ov::element::Type up_dt = ov::element::u4;
    ov::element::Type down_dt = ov::element::u4;
    // Scale (and zp) group counts. G == 1 means per-channel, which is what arms the
    // broadcast_per_channel_scale path; G > 1 is group-wise and sets group_size.
    size_t gate_G = kHidden / kGroupSize;
    size_t up_G = kHidden / kGroupSize;
    size_t down_G = kInter / kGroupSize;
    // element::dynamic == no zp at all (symmetric). Anything else builds real zps whose shape
    // mirrors the corresponding scale.
    ov::element::Type zp_dt = ov::element::dynamic;
};

// Same topology as build_3gemm_bgm_model, with the three GatherMatmul replaced by
// GatherMatmulCompressed. The Parameter stays f32 on purpose: the router MatMul below multiplies
// by an f32 Constant and v0::MatMul requires both operands to share an element type, so making
// the Parameter f16 throws inside this builder before the pass ever runs.
std::shared_ptr<ov::Model> build_3gemm_bgm_compressed_model(const CompressedMoeSpec& spec) {
    using namespace ov;

    auto input =
        std::make_shared<op::v0::Parameter>(element::f32, PartialShape{2, Dimension::dynamic(), kHidden});
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, static_cast<int64_t>(kHidden)}),
        false);
    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(experts_reshape, op::v0::Constant::create(element::i32, Shape{}, {0}));

    auto router_matmul = std::make_shared<op::v0::MatMul>(experts_reshape,
                                                         op::v0::Constant::create(element::f32,
                                                                                  Shape{kE, kHidden},
                                                                                  {1.0f}),
                                                         false,
                                                         true);
    auto router_topk = std::make_shared<op::v11::TopK>(router_matmul,
                                                      op::v0::Constant::create(element::i64, Shape{}, {kTopk}),
                                                      -1,
                                                      op::v11::TopK::Mode::MAX,
                                                      op::v11::TopK::SortType::SORT_VALUES,
                                                      element::i64);
    auto topk_indices = router_topk->output(1);
    auto chosen_experts = router_topk->output(0);

    auto gate_w = int_const(spec.gate_dt, Shape{kE, kInter, kHidden});
    auto up_w = int_const(spec.up_dt, Shape{kE, kInter, kHidden}, 1);
    auto down_w = int_const(spec.down_dt, Shape{kE, kHidden, kInter}, 2);

    auto gate_scale = scale_const(Shape{kE, kInter, spec.gate_G});
    auto up_scale = scale_const(Shape{kE, kInter, spec.up_G});
    auto down_scale = scale_const(Shape{kE, kHidden, spec.down_G});

    const bool has_zp = spec.zp_dt != element::dynamic;
    // 10/11/12 rather than 0..3: a zp of 0 is indistinguishable from the symmetric case, and T6
    // asserts on the expanded values.
    auto gate_zp = has_zp ? int_const(spec.zp_dt, Shape{kE, kInter, spec.gate_G}, 10) : absent_input();
    auto up_zp = has_zp ? int_const(spec.zp_dt, Shape{kE, kInter, spec.up_G}, 11) : absent_input();
    auto down_zp = has_zp ? int_const(spec.zp_dt, Shape{kE, kHidden, spec.down_G}, 12) : absent_input();

    auto bgm_gate = std::make_shared<GatherMatmulCompressed>(unsqueeze,
                                                             gate_w,
                                                             topk_indices,
                                                             absent_input(),
                                                             gate_scale,
                                                             gate_zp);
    auto gate_act = std::make_shared<op::v4::Swish>(bgm_gate);
    auto bgm_up =
        std::make_shared<GatherMatmulCompressed>(unsqueeze, up_w, topk_indices, absent_input(), up_scale, up_zp);
    auto swiglu = std::make_shared<op::v1::Multiply>(gate_act, bgm_up);
    auto bgm_down =
        std::make_shared<GatherMatmulCompressed>(swiglu, down_w, topk_indices, absent_input(), down_scale, down_zp);

    auto router_transpose =
        std::make_shared<op::v1::Transpose>(chosen_experts,
                                            op::v0::Constant::create(element::i64,
                                                                     Shape{2},
                                                                     std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    auto final_mul = std::make_shared<op::v1::Multiply>(bgm_down, router_unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(final_mul,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);
    auto end_reshape = std::make_shared<op::v1::Reshape>(
        reduce_sum,
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{2, -1, static_cast<int64_t>(kHidden)}),
        true);

    return std::make_shared<ov::Model>(ov::OutputVector{end_reshape}, ov::ParameterVector{input});
}

// Runs the 3-GEMM pass and returns the fused node, or nullptr if the pass declined.
std::shared_ptr<MOECompressed> run_3gemm_compressed(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    manager.run_passes(model);
    for (const auto& op : model->get_ops()) {
        if (auto moe = ov::as_type_ptr<MOECompressed>(op)) {
            return moe;
        }
    }
    return nullptr;
}

}  // namespace

// C6. Uniform i8 is the one and only dtype the admission gate regressed: upstream has no gate in
// the compressed branch at all, so an i8 MoE layer fused before this PR and stopped fusing on it.
// After the fix the accepted uniform set is exactly the set the producing pass can emit
// (supported_compressed_weights_types_with_u2 = {u4, i4, i8, u8, u2}).
TEST(Convert3GatherMatmulMoeBlockToMoeOpCompressed, uniform_i8_fuses) {
    CompressedMoeSpec spec;
    spec.gate_dt = spec.up_dt = spec.down_dt = ov::element::i8;
    auto model = build_3gemm_bgm_compressed_model(spec);

    ASSERT_EQ(count_ops_of_type<GatherMatmulCompressed>(model), 3);
    auto moe = run_3gemm_compressed(model);

    ASSERT_NE(moe, nullptr) << "uniform i8 must fuse into MOECompressed";
    EXPECT_EQ(count_ops_of_type<GatherMatmulCompressed>(model), 0);
    const auto& cfg = moe->get_config();
    EXPECT_EQ(cfg.group_size, kGroupSize);
    EXPECT_EQ(cfg.hidden_size, kHidden);
    EXPECT_EQ(cfg.inter_size, kInter);
    EXPECT_EQ(cfg.num_expert, kE);
    EXPECT_EQ(cfg.top_k, kTopk);
    EXPECT_FALSE(cfg.has_zp);
}

// C6. The full admission table. Uniform is allowed for every dtype the producer can emit; mixing
// is allowed only within {u2, u8}, which is what moe_3gemm's per-GEMM dispatch actually decodes.
struct DtypeGateParams {
    ov::element::Type gate_dt;
    ov::element::Type up_dt;
    ov::element::Type down_dt;
    bool should_fuse;
};

class MoeOpFusionDtypeGate : public ::testing::TestWithParam<DtypeGateParams> {};

TEST_P(MoeOpFusionDtypeGate, admits_uniform_and_u2_u8_mixes_only) {
    const auto& p = GetParam();
    CompressedMoeSpec spec;
    spec.gate_dt = p.gate_dt;
    spec.up_dt = p.up_dt;
    spec.down_dt = p.down_dt;

    auto model = build_3gemm_bgm_compressed_model(spec);
    auto moe = run_3gemm_compressed(model);

    if (p.should_fuse) {
        EXPECT_NE(moe, nullptr) << p.gate_dt << "/" << p.up_dt << "/" << p.down_dt << " should fuse";
    } else {
        EXPECT_EQ(moe, nullptr) << p.gate_dt << "/" << p.up_dt << "/" << p.down_dt << " should NOT fuse";
        // Declining must leave the per-GEMM-correct form in place, not a half-rewritten graph.
        EXPECT_EQ(count_ops_of_type<GatherMatmulCompressed>(model), 3);
    }
}

INSTANTIATE_TEST_SUITE_P(
    smoke,
    MoeOpFusionDtypeGate,
    ::testing::Values(
        // Uniform: every dtype the producing pass can emit.
        DtypeGateParams{ov::element::u2, ov::element::u2, ov::element::u2, true},
        DtypeGateParams{ov::element::u4, ov::element::u4, ov::element::u4, true},
        DtypeGateParams{ov::element::i4, ov::element::i4, ov::element::i4, true},
        DtypeGateParams{ov::element::u8, ov::element::u8, ov::element::u8, true},
        DtypeGateParams{ov::element::i8, ov::element::i8, ov::element::i8, true},
        // Mixed within {u2, u8}: the combinations the real NNCF mixed-precision models produce.
        DtypeGateParams{ov::element::u2, ov::element::u8, ov::element::u2, true},
        DtypeGateParams{ov::element::u8, ov::element::u2, ov::element::u8, true},
        DtypeGateParams{ov::element::u2, ov::element::u2, ov::element::u8, true},
        // Mixed involving anything else: no per-GEMM decode exists, so it must stay unfused.
        DtypeGateParams{ov::element::u4, ov::element::u8, ov::element::u8, false},
        DtypeGateParams{ov::element::i8, ov::element::u8, ov::element::i8, false},
        DtypeGateParams{ov::element::u2, ov::element::u4, ov::element::u2, false},
        DtypeGateParams{ov::element::u4, ov::element::i4, ov::element::u4, false},
        // Not producible by the pipeline today; documents that the gate is a whitelist.
        DtypeGateParams{ov::element::f16, ov::element::f16, ov::element::f16, false}));

// C7. A per-channel zp coexisting with a group-wise GEMM sends the zp through
// broadcast_per_channel_scale, whose per-element memcpy uses Type::size() -- which rounds a
// sub-byte type up to a whole byte. Without the guard the expansion reads past the constant and
// interleaves neighbouring channels, and the result still passes MOECompressed validation, so the
// layer fuses with silently wrong zero points. The fix declines instead.
TEST(Convert3GatherMatmulMoeBlockToMoeOpCompressed, subbyte_per_channel_zp_declines) {
    CompressedMoeSpec spec;
    spec.gate_dt = spec.up_dt = spec.down_dt = ov::element::u4;
    spec.zp_dt = ov::element::u4;
    // gate/up per-channel (G == 1) against a group-wise down: group_size comes out of down as 4,
    // so gate/up must be expanded from 1 group to 4 -- that is the call the guard now rejects.
    spec.gate_G = 1;
    spec.up_G = 1;

    auto model = build_3gemm_bgm_compressed_model(spec);
    auto moe = run_3gemm_compressed(model);

    EXPECT_EQ(moe, nullptr) << "a sub-byte per-channel zp must not be expanded";
    EXPECT_EQ(count_ops_of_type<GatherMatmulCompressed>(model), 3);
}

// C7 negative control: the same shape with byte-wide zps is legitimately expandable, and this is
// the only test that asserts the expansion arithmetic rather than just that it was skipped.
// Green on both sides of the fix.
TEST(Convert3GatherMatmulMoeBlockToMoeOpCompressed, u8_per_channel_zp_expands) {
    CompressedMoeSpec spec;
    spec.gate_dt = spec.up_dt = spec.down_dt = ov::element::u8;
    spec.zp_dt = ov::element::u8;
    spec.gate_G = 1;
    spec.up_G = 1;

    auto model = build_3gemm_bgm_compressed_model(spec);
    auto moe = run_3gemm_compressed(model);

    ASSERT_NE(moe, nullptr);
    EXPECT_TRUE(moe->get_config().has_zp);

    const size_t target_G = kHidden / kGroupSize;
    // input 5 is w0_zp; it started as [E, inter, 1] and must come out as [E, inter, target_G]
    // with each channel value repeated across the groups.
    auto gate_zp = ov::as_type_ptr<ov::op::v0::Constant>(moe->input_value(5).get_node_shared_ptr());
    ASSERT_NE(gate_zp, nullptr);
    EXPECT_EQ(gate_zp->get_shape(), (ov::Shape{kE, kInter, target_G}));
    const auto expanded = gate_zp->cast_vector<int32_t>();
    ASSERT_EQ(expanded.size(), kE * kInter * target_G);
    for (size_t ch = 0; ch < kE * kInter; ++ch) {
        const int32_t expected = 10 + static_cast<int32_t>(ch % 4);
        for (size_t g = 0; g < target_G; ++g) {
            EXPECT_EQ(expanded[ch * target_G + g], expected) << "channel " << ch << " group " << g;
        }
    }
}
