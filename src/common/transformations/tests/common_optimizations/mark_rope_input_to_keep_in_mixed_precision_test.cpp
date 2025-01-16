// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/mark_rope_input_to_keep_in_mixed_precision.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/opsets/opset1.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

TEST_F(TransformationTestsF, MarkRopeInputsToKeepInMixedPrecisionTest) {
    /*
    The 2nd/3rd inputs of ROPE is marked as FP32
                Param2  Param3
                  \       /
                   \     /
                  Matmul(FP32)
                      |
                 Transpose(FP32)
                      |
                 Concat(FP32)
                     /   \
                    /     \
    Param1   Cos(FP32)   Sin(FP32)
      \         |          /
       \        |         /
        \       |        /
               ROPE
    */
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 10, 8, 64});
        auto input_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 32, 1});
        auto input_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, 10});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input_a, input_b);
        auto transpose_order =
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{0, 2, 1});
        auto transpose = std::make_shared<ov::opset1::Transpose>(matmul, transpose_order);
        auto concat = std::make_shared<ov::opset1::Concat>(ov::NodeVector{transpose, transpose}, -1);
        auto cos = std::make_shared<ov::opset1::Cos>(concat);
        auto sin = std::make_shared<ov::opset1::Sin>(concat);
        ov::op::internal::RoPE::Config config;
        auto rope =
            std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{input->output(0), cos->output(0), sin->output(0)},
                                                     config);
        model = std::make_shared<ov::Model>(rope, ov::ParameterVector{input, input_a, input_b}, "model");
    }

    manager.register_pass<ov::pass::MarkRopeInputsToKeepInMixedPrecision>();

    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 10, 8, 64});
        auto input_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 32, 1});
        auto input_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, 10});
        auto matmul = std::make_shared<ov::opset1::MatMul>(input_a, input_b);
        auto transpose_order =
            ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{0, 2, 1});
        auto transpose = std::make_shared<ov::opset1::Transpose>(matmul, transpose_order);
        auto concat = std::make_shared<ov::opset1::Concat>(ov::NodeVector{transpose, transpose}, -1);
        auto cos = std::make_shared<ov::opset1::Cos>(concat);
        auto sin = std::make_shared<ov::opset1::Sin>(concat);
        disable_fp16_compression(matmul);
        disable_fp16_compression(transpose);
        disable_fp16_compression(concat);
        disable_fp16_compression(cos);
        disable_fp16_compression(sin);
        ov::op::internal::RoPE::Config config;
        auto rope =
            std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{input->output(0), cos->output(0), sin->output(0)},
                                                     config);
        model_ref = std::make_shared<ov::Model>(rope, ov::ParameterVector{input, input_a, input_b}, "model_ref");
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::RUNTIME_KEYS);
}
