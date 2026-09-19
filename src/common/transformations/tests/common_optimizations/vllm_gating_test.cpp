// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Covers the "vllm_model" rt_info gate around NormalizeVLLMRoPE and
// EraseRedundantConvertPair: other CommonOptimizations callers see neither fire.
#include "transformations/common_optimizations/common_optimizations.hpp"

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/concat.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/relu.hpp"
#include "openvino/op/split.hpp"
#include "openvino/op/variadic_split.hpp"
#include "openvino/opsets/opset1_decl.hpp"
#include "openvino/pass/manager.hpp"

namespace {

size_t count_of_type(const std::shared_ptr<ov::Model>& model, const ov::DiscreteTypeInfo& type) {
    size_t n = 0;
    for (const auto& op : model->get_ops()) {
        if (op->get_type_info().is_castable(type)) {
            ++n;
        }
    }
    return n;
}

std::shared_ptr<ov::Model> build_vllm_rope_model() {
    // cos/sin half-sized to match each split output (see
    // normalize_vllm_rope_test.cpp).
    const ov::Shape full_shape{1, 4, 8};
    const ov::Shape half_shape{1, 4, 4};
    auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, full_shape);
    auto cos = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);
    auto sin = std::make_shared<ov::opset1::Parameter>(ov::element::f32, half_shape);

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {-1});
    auto split = std::make_shared<ov::op::v1::Split>(x, axis, 2);

    auto mul_a = std::make_shared<ov::op::v1::Multiply>(split->output(0), cos);
    auto mul_b = std::make_shared<ov::op::v1::Multiply>(split->output(1), sin);
    auto neg_one = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {-1.0f});
    auto neg = std::make_shared<ov::op::v1::Multiply>(mul_b, neg_one);
    auto sub_branch = std::make_shared<ov::op::v1::Add>(mul_a, neg);

    auto mul_c = std::make_shared<ov::op::v1::Multiply>(split->output(1), cos);
    auto mul_d = std::make_shared<ov::op::v1::Multiply>(split->output(0), sin);
    auto add_branch = std::make_shared<ov::op::v1::Add>(mul_c, mul_d);

    auto concat = std::make_shared<ov::op::v0::Concat>(ov::OutputVector{sub_branch, add_branch}, -1);
    return std::make_shared<ov::Model>(concat, ov::ParameterVector{x, cos, sin});
}

// Round-trip pair, not identity Convert: a generic dead-code pass already
// eliminates identity Converts regardless of this gate.
std::shared_ptr<ov::Model> build_round_trip_convert_model() {
    auto x = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 4});
    auto narrow = std::make_shared<ov::op::v0::Convert>(x, ov::element::bf16);
    auto wide = std::make_shared<ov::op::v0::Convert>(narrow, ov::element::f32);
    auto relu = std::make_shared<ov::opset1::Relu>(wide);
    return std::make_shared<ov::Model>(relu, ov::ParameterVector{x});
}

}  // namespace

TEST(vllm_gating, rope_pass_skipped_without_vllm_model_flag) {
    auto model = build_vllm_rope_model();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::CommonOptimizations>();
    manager.run_passes(model);
    // NormalizeVLLMRoPE not applied: original Split survives, no VariadicSplit
    // was introduced in its place.
    EXPECT_EQ(count_of_type(model, ov::op::v1::Split::get_type_info_static()), 1u);
    EXPECT_EQ(count_of_type(model, ov::op::v1::VariadicSplit::get_type_info_static()), 0u);
}

TEST(vllm_gating, rope_pass_applied_with_vllm_model_flag) {
    auto model = build_vllm_rope_model();
    model->set_rt_info(true, "vllm_model");
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::CommonOptimizations>();
    manager.run_passes(model);
    // NormalizeVLLMRoPE applied: Split replaced by VariadicSplit.
    EXPECT_EQ(count_of_type(model, ov::op::v1::Split::get_type_info_static()), 0u);
    EXPECT_EQ(count_of_type(model, ov::op::v1::VariadicSplit::get_type_info_static()), 1u);
}

TEST(vllm_gating, erase_redundant_convert_pair_skipped_without_vllm_model_flag) {
    auto model = build_round_trip_convert_model();
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::CommonOptimizations>();
    manager.run_passes(model);
    EXPECT_EQ(count_of_type(model, ov::op::v0::Convert::get_type_info_static()), 2u);
}

TEST(vllm_gating, erase_redundant_convert_pair_applied_with_vllm_model_flag) {
    auto model = build_round_trip_convert_model();
    model->set_rt_info(true, "vllm_model");
    ov::pass::Manager manager;
    manager.register_pass<ov::pass::CommonOptimizations>();
    manager.run_passes(model);
    EXPECT_EQ(count_of_type(model, ov::op::v0::Convert::get_type_info_static()), 0u);
}
