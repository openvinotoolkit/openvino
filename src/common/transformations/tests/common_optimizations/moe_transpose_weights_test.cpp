// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/moe_transpose_weights.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <vector>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/clamp.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/matmul.hpp"
#include "openvino/op/minimum.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/reduce_sum.hpp"
#include "openvino/op/reshape.hpp"
#include "openvino/op/scatter_elements_update.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/tile.hpp"
#include "openvino/op/topk.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/unsqueeze.hpp"
#include "openvino/pass/constant_folding.hpp"
#include "transformations/rt_info/decompression.hpp"

using namespace ov;

namespace {

std::shared_ptr<ov::Model> build_moe_2gemm_model(bool use_decompression, bool with_transpose) {
    using namespace ov;

    const size_t batch = 2;
    const Dimension in_dim = Dimension::dynamic();
    const size_t hidden_size = 2048;
    const size_t intermediate_size = 4096;
    const size_t fusion_factor = 2;
    const size_t topk = 2;
    const size_t number_of_experts = 3;

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

    // Pattern model (before transform): weights shape matches non-transposed MatMul
    // Reference model (after transform): weights + Transpose node, MatMul with transpose_b=true
    Output<Node> gate_weight_output;
    auto gate_weights = op::v0::Constant::create(
        element::f32,
        Shape{number_of_experts, hidden_size, intermediate_size * fusion_factor},
        std::vector<float>(number_of_experts * hidden_size * intermediate_size * fusion_factor, 1.0f));
    gate_weights->set_friendly_name("gate_weights");
    gate_weight_output = gate_weights;

    if (use_decompression) {
        auto gate_const = op::v0::Constant::create(
            element::f16,
            Shape{number_of_experts, hidden_size, intermediate_size * fusion_factor},
            std::vector<float16>(number_of_experts * hidden_size * intermediate_size * fusion_factor, 1.0f));
        gate_const->set_friendly_name("gate_weights_storage");

        Output<Node> convert_input = gate_const;
        if (with_transpose) {
            auto order = op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 2, 1});
            auto gate_transpose = std::make_shared<op::v1::Transpose>(convert_input, order);
            gate_transpose->get_rt_info()["postponed_constant"] = true;
            ov::pass::disable_constant_folding(gate_transpose);
            convert_input = gate_transpose;
        }

        auto convert = std::make_shared<op::v0::Convert>(convert_input, element::f32);
        convert->set_friendly_name("gate_convert");
        ov::mark_as_decompression(convert);
        gate_weight_output = convert;
    } else {
        if (with_transpose) {
            auto order = op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 2, 1});
            auto gate_transpose = std::make_shared<op::v1::Transpose>(gate_weight_output, order);
            gate_transpose->get_rt_info()["postponed_constant"] = true;
            ov::pass::disable_constant_folding(gate_transpose);
            gate_weight_output = gate_transpose;
        }
    }

    auto gate_matmul = std::make_shared<op::v0::MatMul>(after_tile_reshape, gate_weight_output, false, with_transpose);

    auto gate_bias = std::make_shared<op::v1::Add>(
        gate_matmul,
        op::v0::Constant::create(element::f32,
                                 Shape{number_of_experts, 1, intermediate_size * fusion_factor},
                                 std::vector<float>(number_of_experts * intermediate_size * fusion_factor, 0.0f)));

    auto slice1 = std::make_shared<op::v8::Slice>(
        gate_bias,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(number_of_experts),
                                                      static_cast<int64_t>(batch),
                                                      static_cast<int64_t>(intermediate_size * fusion_factor)}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto clamp = std::make_shared<op::v0::Clamp>(slice1, -5.0f, 5.0f);
    auto add1 = std::make_shared<op::v1::Add>(clamp, op::v0::Constant::create(element::f32, Shape{1}, {1.0f}));

    auto slice2 = std::make_shared<op::v8::Slice>(
        gate_bias,
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 0, 0}),
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(number_of_experts),
                                                      static_cast<int64_t>(batch),
                                                      static_cast<int64_t>(intermediate_size * fusion_factor)}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{1, 1, 2}),
        op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 1, 2}));
    auto minimum1 =
        std::make_shared<op::v1::Minimum>(slice2, op::v0::Constant::create(element::f32, Shape{1}, {10.0f}));
    auto swish_beta = op::v0::Constant::create(element::f32, Shape{}, std::vector<float>{1.7f});
    auto swish = std::make_shared<op::v4::Swish>(minimum1, swish_beta);

    auto multiply2 = std::make_shared<op::v1::Multiply>(add1, swish);

    Output<Node> down_weight_output;
    if (use_decompression) {
        auto down_const =
            op::v0::Constant::create(element::f16,
                                     Shape{number_of_experts, intermediate_size, hidden_size},
                                     std::vector<float16>(number_of_experts * intermediate_size * hidden_size, 2));
        down_const->set_friendly_name("down_weights_storage");

        Output<Node> convert_input = down_const;
        if (with_transpose) {
            auto order = op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 2, 1});
            auto down_transpose = std::make_shared<op::v1::Transpose>(convert_input, order);
            down_transpose->get_rt_info()["postponed_constant"] = true;
            ov::pass::disable_constant_folding(down_transpose);
            convert_input = down_transpose;
        }

        auto convert = std::make_shared<op::v0::Convert>(convert_input, element::f32);
        convert->set_friendly_name("down_convert");
        ov::mark_as_decompression(convert);
        down_weight_output = convert;
    } else {
        auto down_const =
            op::v0::Constant::create(element::f32,
                                     Shape{number_of_experts, intermediate_size, hidden_size},
                                     std::vector<float>(number_of_experts * intermediate_size * hidden_size, 0.5f));
        down_const->set_friendly_name("down_weights");

        down_weight_output = down_const;
        if (with_transpose) {
            auto order = op::v0::Constant::create(element::i64, Shape{3}, std::vector<int64_t>{0, 2, 1});
            auto down_transpose = std::make_shared<op::v1::Transpose>(down_weight_output, order);
            down_transpose->get_rt_info()["postponed_constant"] = true;
            ov::pass::disable_constant_folding(down_transpose);
            down_weight_output = down_transpose;
        }
    }

    auto down_matmul = std::make_shared<op::v0::MatMul>(multiply2, down_weight_output, false, with_transpose);

    auto down_bias = std::make_shared<op::v1::Add>(
        down_matmul,
        op::v0::Constant::create(element::f32,
                                 Shape{number_of_experts, 1, hidden_size},
                                 std::vector<float>(number_of_experts * hidden_size, 0.0f)));
    auto end_reshape = std::make_shared<op::v1::Reshape>(
        down_bias,
        op::v0::Constant::create(element::i64,
                                 Shape{4},
                                 std::vector<int64_t>{static_cast<int64_t>(number_of_experts),
                                                      static_cast<int64_t>(batch),
                                                      -1,
                                                      static_cast<int64_t>(hidden_size)}),
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
    auto router_topk_values = router_topk->output(0);
    auto router_topk_indices = router_topk->output(1);

    auto scatter = std::make_shared<op::v12::ScatterElementsUpdate>(
        router_topk_values,
        router_topk_indices,
        op::v0::Constant::create(element::f32, Shape{batch, topk}, {0.0f}),
        op::v0::Constant::create(element::i64, Shape{1}, {1}));
    auto router_transpose = std::make_shared<op::v1::Transpose>(
        scatter,
        op::v0::Constant::create(element::i64, Shape{2}, std::vector<int64_t>{1, 0}));
    auto router_reshape = std::make_shared<op::v1::Reshape>(
        router_transpose,
        op::v0::Constant::create(element::i64,
                                 Shape{3},
                                 std::vector<int64_t>{static_cast<int64_t>(number_of_experts), 1, -1}),
        true);
    auto unsqueeze =
        std::make_shared<op::v0::Unsqueeze>(router_reshape,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{-1}));

    auto mul3 = std::make_shared<op::v1::Multiply>(end_reshape, unsqueeze);
    auto reduce_sum =
        std::make_shared<op::v1::ReduceSum>(mul3,
                                            op::v0::Constant::create(element::i64, Shape{1}, std::vector<int64_t>{0}),
                                            false);

    return std::make_shared<ov::Model>(OutputVector{reduce_sum}, ParameterVector{input});
}

std::shared_ptr<ov::Model> build_moe_2gemm_pattern_model(bool use_decompression) {
    return build_moe_2gemm_model(use_decompression, false);
}

std::shared_ptr<ov::Model> build_moe_2gemm_reference_model(bool use_decompression) {
    return build_moe_2gemm_model(use_decompression, true);
}

}  // namespace

TEST_F(TransformationTestsF, VectorizedMOE2GEMMTransposeWeightsConstantWeights) {
    model = build_moe_2gemm_pattern_model(/*use_decompression=*/false);
    model_ref = build_moe_2gemm_reference_model(/*use_decompression=*/false);
    manager.register_pass<ov::pass::VectorizedMOE2GEMMTransposeWeights>();
}

TEST_F(TransformationTestsF, VectorizedMOE2GEMMTransposeWeightsDecompressionWeights) {
    model = build_moe_2gemm_pattern_model(/*use_decompression=*/true);
    model_ref = build_moe_2gemm_reference_model(/*use_decompression=*/true);
    manager.register_pass<ov::pass::VectorizedMOE2GEMMTransposeWeights>();
}
