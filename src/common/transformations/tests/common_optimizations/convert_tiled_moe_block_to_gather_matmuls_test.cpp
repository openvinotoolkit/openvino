// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/convert_tiled_moe_block_to_gather_matmuls.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <openvino/core/model.hpp>
#include <openvino/op/add.hpp>
#include <openvino/op/clamp.hpp>
#include <openvino/op/constant.hpp>
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
#include "transformations/common_optimizations/moe_op_fusion.hpp"

using GatherMatmul = ov::op::internal::GatherMatmul;

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

inline std::shared_ptr<ov::Model> build_3gemm_bgm_model() {
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
    auto swish = std::make_shared<op::v4::Swish>(bgm_gate);
    auto bgm_up = std::make_shared<GatherMatmul>(unsqueeze, up_w, topk_indices);
    auto swiglu = std::make_shared<op::v1::Multiply>(swish, bgm_up);
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

inline std::shared_ptr<ov::Model> build_3gemm_bgm_to_moe_reference_model() {
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
    auto chosen_experts = router_topk->output(0);

    // Compact routing (stays as-is, becomes MOE input 1)
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Weights
    auto gate_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto up_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});

    // MOE op with compact routing
    ov::OutputVector moe_inputs = {input, router_unsqueeze, topk_indices, gate_w, up_w, down_w};
    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM3_SWIGLU;
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

    // Compact routing (becomes MOE input 1)
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        chosen_experts,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_transpose, op::v0::Constant::create(element::i32, Shape{}, {-1}));

    // Weights
    auto gate_up_w = op::v0::Constant::create(element::f32,
                                              Shape{number_of_experts, intermediate_size * fusion_factor, hidden_size},
                                              {1.0f});
    auto gate_up_bias =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f});
    auto down_w =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, hidden_size, intermediate_size}, {1.0f});
    auto down_bias = op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f});

    ov::OutputVector moe_inputs = {input, router_unsqueeze, topk_indices, gate_up_w, gate_up_bias, down_w, down_bias};

    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;
    config.expert_alpha = expert_alpha;
    config.expert_beta = expert_beta;

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

TEST_F(TransformationTestsF, Convert3GatherMatmulMoeBlockToMoeOp_no_fusion_on_2gemm) {
    model = build_2gemm_bgm_model();
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    // No model_ref — should not fuse
}

TEST_F(TransformationTestsF, Convert2GatherMatmulMoeBlockToMoeOp_basic) {
    model = build_2gemm_bgm_model();
    manager.register_pass<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>();
    model_ref = build_2gemm_bgm_to_moe_reference_model();
}

TEST_F(TransformationTestsF, Convert2GatherMatmulMoeBlockToMoeOp_no_fusion_on_3gemm) {
    model = build_3gemm_bgm_model();
    manager.register_pass<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>();
    // No model_ref — should not fuse
}

// ============================================================================
// End-to-end pipeline test: IR → BGM → MOE (steps 1+2)
// Verifies that both passes compose correctly and produce a MOE node.
// ============================================================================

TEST(FuseEndToEnd, EndToEnd_3GEMM_IR_to_BGM_to_MOE) {
    auto model = build_3gemm_moe_pattern_model();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertTiledMoeBlockTo3GatherMatmuls>();
    manager.register_pass<ov::pass::Convert3GatherMatmulMoeBlockToMoeOp>();
    manager.run_passes(model);

    // Verify that a MOE node was produced
    bool found_moe = false;
    for (const auto& op : model->get_ops()) {
        if (ov::as_type_ptr<ov::op::internal::MOE>(op)) {
            found_moe = true;
            EXPECT_EQ(op->get_input_size(), 6u);  // hidden, routing, topk, gate_w, up_w, down_w
            break;
        }
    }
    EXPECT_TRUE(found_moe) << "Expected MOE node after IR→BGM→MOE pipeline";
}

TEST(FuseEndToEnd, EndToEnd_2GEMM_IR_to_BGM_to_MOE) {
    auto model = build_2gemm_moe_pattern_model();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::ConvertTiledMoeBlockTo2GatherMatmuls>();
    manager.register_pass<ov::pass::Convert2GatherMatmulMoeBlockToMoeOp>();
    manager.run_passes(model);

    bool found_moe = false;
    for (const auto& op : model->get_ops()) {
        if (auto moe = ov::as_type_ptr<ov::op::internal::MOE>(op)) {
            found_moe = true;
            EXPECT_EQ(op->get_input_size(), 7u);  // hidden, routing, topk, gate_up_w, gate_up_bias, down_w, down_bias
            EXPECT_EQ(moe->get_config().expert_type, ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP);
            break;
        }
    }
    EXPECT_TRUE(found_moe) << "Expected MOE node after IR→BGM→MOE pipeline";
}
