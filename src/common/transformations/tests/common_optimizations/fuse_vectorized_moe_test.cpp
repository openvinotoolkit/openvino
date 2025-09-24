// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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
#include <openvino/pass/serialize.hpp>
#include <openvino/pass/visualize_tree.hpp>
#include <vector>

#include "common_test_utils/graph_comparator.hpp"
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/node_vector.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/runtime/core.hpp"
#include "ov_ops/type_relaxed.hpp"
#include "transformations/common_optimizations/matmul_experts_fusion.hpp"
#include "transformations/utils/gen_pattern.hpp"

inline std::shared_ptr<ov::Model> build_moe_pattern_model() {
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
                                 Shape{number_of_experts, hidden_size, intermediate_size * fusion_factor},
                                 {1.0f}));
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
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f}));

    auto down_proj_add = std::make_shared<op::v1::Add>(
        down_proj_matmul,
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f}));

    auto end_reshape = std::make_shared<op::v1::Reshape>(
        down_proj_add,
        op::v0::Constant::create(element::i64,
                                 Shape{4},
                                 std::vector<int64_t>{number_of_experts, batch, -1, hidden_size}),
        false);

    // Router subgraph used to test correctness of routing weights connection
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
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));

    auto mul3 = std::make_shared<op::v1::Multiply>(end_reshape, unsqueeze_routing_weights);

    // ReduceSum - final node of the MOE pattern to be fused
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(mul3,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            true);

    return std::make_shared<ov::Model>(ov::OutputVector{reduce_sum}, ov::ParameterVector{input});
}

inline std::shared_ptr<ov::Model> build_fused_moe_reference_model() {
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

    // Begin of Router subgraph (not fused, but valuable for testing)
    auto experts_reshape = std::make_shared<op::v1::Reshape>(
        input,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{-1, hidden_size}),
        false);

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
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{1}));
    // End of Router subgraph

    // Expert MatMuls weights fused into MOE
    auto w0_weight = op::v0::Constant::create(element::f32,
                                              Shape{number_of_experts, hidden_size, intermediate_size * fusion_factor},
                                              {1.0f});
    auto w0_bias =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, intermediate_size * fusion_factor}, {0.0f});
    auto w1_weight =
        op::v0::Constant::create(element::f32, Shape{number_of_experts, intermediate_size, hidden_size}, {1.0f});
    auto w1_bias = op::v0::Constant::create(element::f32, Shape{number_of_experts, 1, hidden_size}, {1.0f});

    ov::OutputVector moe_inputs =
        {input, unsqueeze_routing_weights, router_topk_indices, w0_weight, w0_bias, w1_weight, w1_bias};

    ov::op::internal::MOE::Config config;
    config.expert_type = ov::op::internal::MOE::Expert_type::GEMM2_BIAS_SWIGLU_CLAMP;
    config.expert_alpha = expert_alpha;
    config.expert_beta = expert_beta;

    auto moe = std::make_shared<ov::op::internal::MOE>(moe_inputs, config);
    return std::make_shared<ov::Model>(ov::OutputVector{moe}, ov::ParameterVector{input});
}

TEST_F(TransformationTestsF, FuseVectorizedMOE_basic) {
    model = build_moe_pattern_model();
    manager.register_pass<ov::pass::FuseVectorizedMOE>();
    model_ref = build_fused_moe_reference_model();
}
