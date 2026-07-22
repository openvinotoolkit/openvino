// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/convert_precision.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/rt_info/disable_precision_conversion.hpp>

#include "openvino/op/constant.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "plugin/transformations/disable_fp16_comp_cumsum_sin_gen.hpp"

using namespace testing;
using namespace ov::intel_gpu;

// Friendly names used to look up the matched nodes after the pass has run.
namespace {
const std::string name_cumsum_input = "cumsum_input";
const std::string name_cumsum = "cumsum";
const std::string name_mul1 = "mul1";
const std::string name_transpose2 = "transpose2";
const std::string name_mul2 = "mul2";
const std::string name_interpolate = "interpolate";
const std::string name_transpose3 = "transpose3";
const std::string name_sin = "sin";

// Build the full chain matched by DisableFP16CompCumSumSinGen:
//   producer -> CumSum -> Multiply -> Transpose -> Multiply -> Interpolate -> Transpose -> Sin
std::shared_ptr<ov::Model> create_model_to_match() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    // Extra Multiply so the test can verify the CumSum producer gets marked.
    auto producer_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {1.0f});
    auto cumsum_input = std::make_shared<ov::op::v1::Multiply>(input, producer_const);
    cumsum_input->set_friendly_name(name_cumsum_input);

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto cumsum = std::make_shared<ov::op::v0::CumSum>(cumsum_input, axis);
    cumsum->set_friendly_name(name_cumsum);

    auto mul1_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {6.2832f});
    auto mul1 = std::make_shared<ov::op::v1::Multiply>(cumsum, mul1_const);
    mul1->set_friendly_name(name_mul1);

    auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose2 = std::make_shared<ov::op::v1::Transpose>(mul1, order2);
    transpose2->set_friendly_name(name_transpose2);

    auto mul2_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{32, 1}, {1.0f});
    auto mul2 = std::make_shared<ov::op::v1::Multiply>(transpose2, mul2_const);
    mul2->set_friendly_name(name_mul2);

    ov::op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {64, 1});
    auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 1.0f});
    auto interpolate = std::make_shared<ov::op::v4::Interpolate>(mul2, target_shape, scales, attrs);
    interpolate->set_friendly_name(name_interpolate);

    auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose3 = std::make_shared<ov::op::v1::Transpose>(interpolate, order3);
    transpose3->set_friendly_name(name_transpose3);

    auto sin = std::make_shared<ov::op::v0::Sin>(transpose3);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

// A bare Sin (no upstream CumSum) — must not be matched.
std::shared_ptr<ov::Model> create_model_not_to_match() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    auto mul_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {1.0f});
    auto mul = std::make_shared<ov::op::v1::Multiply>(input, mul_const);
    mul->set_friendly_name(name_mul1);

    auto sin = std::make_shared<ov::op::v0::Sin>(mul);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

// Same chain but without the second Multiply between Transpose_2 and
// Interpolate — must not match.
std::shared_ptr<ov::Model> create_model_missing_mul2() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto cumsum = std::make_shared<ov::op::v0::CumSum>(input, axis);
    cumsum->set_friendly_name(name_cumsum);

    auto mul1_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {6.2832f});
    auto mul1 = std::make_shared<ov::op::v1::Multiply>(cumsum, mul1_const);
    mul1->set_friendly_name(name_mul1);

    auto order2 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose2 = std::make_shared<ov::op::v1::Transpose>(mul1, order2);
    transpose2->set_friendly_name(name_transpose2);

    // NOTE: no second Multiply here.

    ov::op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;
    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {64, 1});
    auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 1.0f});
    auto interpolate = std::make_shared<ov::op::v4::Interpolate>(transpose2, target_shape, scales, attrs);
    interpolate->set_friendly_name(name_interpolate);

    auto order3 = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose3 = std::make_shared<ov::op::v1::Transpose>(interpolate, order3);
    transpose3->set_friendly_name(name_transpose3);

    auto sin = std::make_shared<ov::op::v0::Sin>(transpose3);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

void run_test(const std::shared_ptr<ov::Model>& model,
              const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16CompCumSumSinGen>();

    precisions_map fp_convert_precision_map = {{ov::element::f32, ov::element::f16}};
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map);

    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it == expected_fp16_disabled_status.end())
            continue;
        if (it->second) {
            ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 compression is not disabled for node: " << op->get_friendly_name();
        } else {
            ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f16))
                << "FP16 compression is unexpectedly disabled for node: " << op->get_friendly_name();
        }
    }
}
}  // namespace

TEST(TransformationTests, DisableFP16CompCumSumSinGen_Positive) {
    auto model = create_model_to_match();
    // The pass marks the full 7-node chain plus the producer feeding CumSum.
    std::unordered_map<std::string, bool> expected_status = {
        {name_cumsum_input, true},
        {name_cumsum, true},
        {name_mul1, true},
        {name_transpose2, true},
        {name_mul2, true},
        {name_interpolate, true},
        {name_transpose3, true},
        {name_sin, true},
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompCumSumSinGen_NoCumSumUpstream_NoOp) {
    auto model = create_model_not_to_match();
    std::unordered_map<std::string, bool> expected_status = {
        {name_mul1, false},
        {name_sin, false},
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompCumSumSinGen_MissingSecondMultiply_NoOp) {
    auto model = create_model_missing_mul2();
    // No second Multiply — pattern must not match.
    std::unordered_map<std::string, bool> expected_status = {
        {name_cumsum, false},
        {name_mul1, false},
        {name_transpose2, false},
        {name_interpolate, false},
        {name_transpose3, false},
        {name_sin, false},
    };
    run_test(model, expected_status);
}
