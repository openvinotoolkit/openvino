// Copyright (C) 2018-2024 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include <gtest/gtest.h>
#include "openvino/opsets/opset1.hpp"
#include "ov_ops/rotary_positional_embeddings.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/cpu_opset/common/pass/markup_rope_inputs.hpp"
#include "transformations/rt_info/disable_fp16_compression.hpp"

TEST(TransformationTests, MarkUpRopeInputsTest) {
    ov::ParameterVector params;
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 10, 8, 64});
    auto input_a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 32, 1});
    auto input_b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 1, 10});
    auto matmul = std::make_shared<ov::opset1::MatMul>(input_a, input_b);
    auto transpose_order = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{3}, std::vector<int32_t>{0, 2, 1});
    auto transpose = std::make_shared<ov::opset1::Transpose>(matmul, transpose_order);
    auto concat = std::make_shared<ov::opset1::Concat>(ov::NodeVector{transpose, transpose}, -1);
    auto cos = std::make_shared<ov::opset1::Cos>(concat);
    auto sin = std::make_shared<ov::opset1::Sin>(concat);
    ov::op::internal::RoPE::Config config;
    auto rope =
        std::make_shared<ov::op::internal::RoPE>(ov::OutputVector{input->output(0), cos->output(0), sin->output(0)},
                                                 config);
    auto function = std::make_shared<ov::Model>(rope, ov::ParameterVector{input, input_a, input_b}, "Subgraph");
    ov::pass::Manager m;
    m.register_pass<ov::intel_cpu::MarkUpRopeInputs>();
    m.run_passes(function);
    for (const auto& node : function->get_ordered_ops()) {
        if (ov::is_type<ov::opset1::MatMul>(node) || ov::is_type<ov::opset1::Cos>(node) || ov::is_type<ov::opset1::Sin>(node)) {
            ASSERT_TRUE(ov::fp16_compression_is_disabled(node));
        }
    }
}
