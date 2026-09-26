// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Covers NormalizeVLLMMLP's two top-level input forms and its no-op case;
// not the narrow-Convert or rank-2-to-rank-3 sub-cases.
#include "transformations/common_optimizations/normalize_vllm_mlp.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/slice.hpp"
#include "openvino/op/swish.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/manager.hpp"

namespace {
std::shared_ptr<ov::op::v8::Slice> make_slice(const ov::Output<ov::Node>& data, int64_t start, int64_t stop) {
    auto starts = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {start});
    auto stops = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {stop});
    auto steps = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {1});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
    return std::make_shared<ov::op::v8::Slice>(data, starts, stops, steps, axes);
}
}  // namespace

// Branch A: two Slice ops on the same source feeding Swish(gate)*up -- must
// be rewritten to one VariadicSplit(axis=-1, lengths=[half,half]).
TEST_F(TransformationTestsF, NormalizeVLLMMLPRewritesTwoSliceForm) {
    const ov::Shape shape{1, 4, 8};
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto gate_slice = make_slice(src, 0, 4);
        auto up_slice = make_slice(src, 4, 8);
        auto swish = std::make_shared<ov::op::v4::Swish>(gate_slice);
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, up_slice);
        model = std::make_shared<ov::Model>(mul, ov::ParameterVector{src}, "model");
    }

    manager.register_pass<ov::pass::NormalizeVLLMMLP>();

    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        auto lengths = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {4, 4});
        auto vsplit = std::make_shared<ov::op::v1::VariadicSplit>(src, axis, lengths);
        auto swish = std::make_shared<ov::op::v4::Swish>(vsplit->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, vsplit->output(1));
        model_ref = std::make_shared<ov::Model>(mul, ov::ParameterVector{src}, "model_ref");
    }
}

// Branch B: VariadicSplit already present but with i64 lengths and a
// positive axis -- canonicalized to i32 lengths [2] and literal axis -1.
TEST_F(TransformationTestsF, NormalizeVLLMMLPCanonicalizesVariadicSplit) {
    const ov::Shape shape{4, 8};
    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
        auto lengths = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {4, 4});
        auto vsplit = std::make_shared<ov::op::v1::VariadicSplit>(src, axis, lengths);
        auto swish = std::make_shared<ov::op::v4::Swish>(vsplit->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, vsplit->output(1));
        model = std::make_shared<ov::Model>(mul, ov::ParameterVector{src}, "model");
    }

    manager.register_pass<ov::pass::NormalizeVLLMMLP>();

    {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        auto lengths = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {4, 4});
        auto vsplit = std::make_shared<ov::op::v1::VariadicSplit>(src, axis, lengths);
        auto swish = std::make_shared<ov::op::v4::Swish>(vsplit->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, vsplit->output(1));
        model_ref = std::make_shared<ov::Model>(mul, ov::ParameterVector{src}, "model_ref");
    }
}

// Already-canonical VariadicSplit (i32 lengths [2], literal axis -1, no
// wedged Convert) -- pass must be a no-op.
TEST_F(TransformationTestsF, NormalizeVLLMMLPNoOpOnAlreadyCanonicalForm) {
    const ov::Shape shape{1, 4, 8};
    auto build = [&]() {
        auto src = std::make_shared<ov::opset1::Parameter>(ov::element::f32, shape);
        auto axis = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{}, {-1});
        auto lengths = ov::op::v0::Constant::create(ov::element::i32, ov::Shape{2}, {4, 4});
        auto vsplit = std::make_shared<ov::op::v1::VariadicSplit>(src, axis, lengths);
        auto swish = std::make_shared<ov::op::v4::Swish>(vsplit->output(0));
        auto mul = std::make_shared<ov::op::v1::Multiply>(swish, vsplit->output(1));
        return std::make_shared<ov::Model>(mul, ov::ParameterVector{src});
    };
    model = build();
    model->set_friendly_name("model");
    manager.register_pass<ov::pass::NormalizeVLLMMLP>();
    model_ref = build();
    model_ref->set_friendly_name("model_ref");
}
