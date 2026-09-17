// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/normalize_vllm_rope.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/manager.hpp"

// Builds vLLM's split/multiply/add neox-style RoPE lowering and checks it
// rewrites into the concat/multiply/add form RoPEFusionGPTNEOX expects.
TEST_F(TransformationTestsF, NormalizeVLLMRoPERewritesToFusableForm) {
    // The pass's copy_runtime_info() call omits some new nodes, a
    // pre-existing gap other tests in this directory also work around.
    disable_rt_info_check();
    const ov::Shape full_shape{1, 4, 8};
    // cos/sin half-sized (matching each split output) exercises the pass's
    // cos/sin duplication path back to full size.
    const ov::Shape half_shape{1, 4, 4};
    {
        auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, full_shape);
        auto cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);
        auto sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);

        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
        auto split = std::make_shared<ov::op::v1::Split>(x, axis, 2);

        auto mul_a = std::make_shared<ov::op::v1::Multiply>(split->output(0), cos);  // x1*cos
        auto mul_b = std::make_shared<ov::op::v1::Multiply>(split->output(1), sin);  // x2*sin
        auto neg_one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
        auto neg = std::make_shared<ov::op::v1::Multiply>(mul_b, neg_one);
        auto sub_branch = std::make_shared<ov::op::v1::Add>(mul_a, neg);

        auto mul_c = std::make_shared<ov::op::v1::Multiply>(split->output(1), cos);  // x2*cos
        auto mul_d = std::make_shared<ov::op::v1::Multiply>(split->output(0), sin);  // x1*sin
        auto add_branch = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);

        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub_branch, add_branch}, -1);
        model = std::make_shared<ov::Model>(concat, ov::ParameterVector{x, cos, sin}, "model");
    }

    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();

    {
        auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, full_shape);
        auto cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);
        auto sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);
        auto cos_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{cos, cos}, -1);
        auto sin_full = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sin, sin}, -1);

        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {2});
        auto lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {4, 4});
        auto vsplit = std::make_shared<ov::op::v1::VariadicSplit>(x, axis, lengths);

        auto neg_one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
        auto x2_neg = std::make_shared<ov::op::v1::Multiply>(vsplit->output(1), neg_one);
        auto x_rot = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{x2_neg, vsplit->output(0)}, -1);
        auto x_cos = std::make_shared<ov::op::v1::Multiply>(x, cos_full);
        auto xrot_sin = std::make_shared<ov::op::v1::Multiply>(x_rot, sin_full);
        auto out = std::make_shared<ov::op::v1::Add>(x_cos, xrot_sin);
        model_ref = std::make_shared<ov::Model>(out, ov::ParameterVector{x, cos, sin}, "model_ref");
    }
}

// Not vLLM's is_neox_style lowering (concat has 3 inputs, not 2) -- must be
// left untouched.
TEST_F(TransformationTestsF, NormalizeVLLMRoPENoMatchOnUnrelatedConcat) {
    const ov::Shape shape{1, 4, 8};
    {
        auto a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto c = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{a, b, c}, -1);
        model = std::make_shared<ov::Model>(concat, ov::ParameterVector{a, b, c}, "model");
    }
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
    {
        auto a = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto b = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto c = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{a, b, c}, -1);
        model_ref = std::make_shared<ov::Model>(concat, ov::ParameterVector{a, b, c}, "model_ref");
    }
}
