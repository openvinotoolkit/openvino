// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
#include "transformations/common_optimizations/normalize_vllm_rope.hpp"

#include <functional>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/squeeze.hpp"
#include "openvino/op/unsqueeze.hpp"
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

namespace {
using View = std::function<ov::Output<ov::Node>(const ov::Output<ov::Node>&)>;

// vLLM's neox lowering: Concat(x1*cos - x2*sin, x2*cos + x1*sin), with the split, cos/sin shapes and concat axis
// given by the caller. `view`, if set, is applied to both halves before the multiplies.
std::shared_ptr<ov::Model> build_vllm_rope(const ov::PartialShape& x_shape,
                                           const ov::PartialShape& cos_shape,
                                           const ov::PartialShape& sin_shape,
                                           const std::vector<int64_t>& split_lengths,
                                           int64_t concat_axis,
                                           bool lengths_as_input = false,
                                           const View& view = nullptr) {
    auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, x_shape);
    auto cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, cos_shape);
    auto sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, sin_shape);
    ov::ParameterVector params{x, cos, sin};
    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    std::shared_ptr<ov::Node> lengths;
    if (lengths_as_input) {
        auto p = std::make_shared<ov::opset1::Parameter>(ov::element::i64, ov::Shape{2});
        params.push_back(p);
        lengths = p;
    } else {
        lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, split_lengths);
    }
    auto split = std::make_shared<ov::op::v1::VariadicSplit>(x, axis, lengths);
    auto x1 = view ? view(split->output(0)) : split->output(0);
    auto x2 = view ? view(split->output(1)) : split->output(1);
    auto mul_a = std::make_shared<ov::op::v1::Multiply>(x1, cos);
    auto mul_b = std::make_shared<ov::op::v1::Multiply>(x2, sin);
    auto neg_one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto neg = std::make_shared<ov::op::v1::Multiply>(mul_b, neg_one);
    auto sub_branch = std::make_shared<ov::op::v1::Add>(mul_a, neg);
    auto mul_c = std::make_shared<ov::op::v1::Multiply>(x2, cos);
    auto mul_d = std::make_shared<ov::op::v1::Multiply>(x1, sin);
    auto add_branch = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);
    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub_branch, add_branch}, concat_axis);
    return std::make_shared<ov::Model>(concat, params, "model");
}
}  // namespace

// A split into unequal halves is not neox RoPE; rewriting it as [D/2, D/2] would change the output shape.
TEST_F(TransformationTestsF, NormalizeVLLMRoPENoMatchOnUnequalSplit) {
    model = build_vllm_rope(ov::PartialShape{5, 4, 8}, ov::PartialShape{}, ov::PartialShape{}, {7, 1}, -1);
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
}

// Split lengths that are not constant can't be proven equal.
TEST_F(TransformationTestsF, NormalizeVLLMRoPENoMatchOnNonConstantSplitLengths) {
    model = build_vllm_rope(ov::PartialShape{5, 4, 8}, ov::PartialShape{}, ov::PartialShape{}, {}, -1, true);
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
}

// The halves joined on another axis is not RoPE either.
TEST_F(TransformationTestsF, NormalizeVLLMRoPENoMatchOnNonLastConcatAxis) {
    model = build_vllm_rope(ov::PartialShape{5, 4, 8}, ov::PartialShape{5, 1, 4}, ov::PartialShape{5, 1, 4}, {4, 4}, 0);
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
}

// A cos width that isn't known statically can't be classified as half width or broadcast, so it's left alone.
TEST_F(TransformationTestsF, NormalizeVLLMRoPENoMatchOnDynamicCosWidth) {
    model =
        build_vllm_rope(ov::PartialShape{5, 4, 8}, ov::PartialShape{5, 1, -1}, ov::PartialShape{5, 1, 4}, {4, 4}, -1);
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
}

// sin is checked as well as cos: a half-width cos with a broadcast sin is not vLLM's form, and the CPU RoPE fusion
// would treat the broadcast sin as a full table.
TEST_F(TransformationTestsF, NormalizeVLLMRoPENoMatchOnSinShapeDifferentFromCos) {
    model =
        build_vllm_rope(ov::PartialShape{5, 4, 8}, ov::PartialShape{5, 1, 4}, ov::PartialShape{5, 1, 1}, {4, 4}, -1);
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
}

// Broadcast (width 1) cos and sin of the same shape are left alone too.
TEST_F(TransformationTestsF, NormalizeVLLMRoPENoMatchOnBroadcastCosSin) {
    model =
        build_vllm_rope(ov::PartialShape{5, 4, 8}, ov::PartialShape{5, 1, 1}, ov::PartialShape{5, 1, 1}, {4, 4}, -1);
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
}

// Halves squeezed to [2, 2] broadcast against cos/sin [2, 2, 2] with the token axis on cos axis 1, while the rewrite
// x[2, 1, 4] * cat(cos) would put it on axis 0: same output shape, different values, so it must not match.
TEST_F(TransformationTestsF, NormalizeVLLMRoPENoMatchOnSqueezedSplitOutputs) {
    auto squeeze = [](const ov::Output<ov::Node>& o) -> ov::Output<ov::Node> {
        return std::make_shared<ov::op::v0::Squeeze>(o, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}));
    };
    model = build_vllm_rope(ov::PartialShape{2, 1, 4},
                            ov::PartialShape{2, 2, 2},
                            ov::PartialShape{2, 2, 2},
                            {2, 2},
                            -1,
                            false,
                            squeeze);
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
}

// Unsqueezed halves: the rewrite x[3, 4] * cat(cos)[3, 5, 4] would not even broadcast, so the callback must reject it
// rather than throw.
TEST_F(TransformationTestsF, NormalizeVLLMRoPENoMatchOnUnsqueezedSplitOutputs) {
    auto unsqueeze = [](const ov::Output<ov::Node>& o) -> ov::Output<ov::Node> {
        return std::make_shared<ov::op::v0::Unsqueeze>(o, ov::op::v0::Constant::create(ov::element::i64, {1}, {1}));
    };
    model = build_vllm_rope(ov::PartialShape{3, 4},
                            ov::PartialShape{3, 5, 2},
                            ov::PartialShape{3, 5, 2},
                            {2, 2},
                            -1,
                            false,
                            unsqueeze);
    manager.register_pass<ov::pass::NormalizeVLLMRoPE>();
}
