// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/pass/manager.hpp>
#include <transformations/utils/utils.hpp>
#include <transformations/convert_precision.hpp>

#include "plugin/transformations/disable_fp16_comp_sin_gen.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/convert.hpp"
#include "openvino/op/add.hpp"

using namespace testing;
using namespace ov::intel_gpu;

static const std::string name_multiply = "multiply";
static const std::string name_interpolate = "interpolate";
static const std::string name_transpose = "transpose";
static const std::string name_sin = "sin";

// This model creates the exact pattern that DisableFP16CompSinGen is looking for.
// multiply - interpolate - transpose - sin
static std::shared_ptr<ov::Model> create_model_to_match() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    // Pattern part 1: multiply
    auto mul_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {1.0f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(input, mul_const);
    multiply->set_friendly_name(name_multiply);

    // Pattern part 2: interpolate
    ov::op::v4::Interpolate::InterpolateAttrs attrs;
    attrs.mode = ov::op::v4::Interpolate::InterpolateMode::NEAREST;
    attrs.shape_calculation_mode = ov::op::v4::Interpolate::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ov::op::v4::Interpolate::NearestMode::ROUND_PREFER_FLOOR;

    auto target_shape = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 64});
    auto scales = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {1.0f, 2.0f});
    auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
    auto interpolate = std::make_shared<ov::op::v4::Interpolate>(multiply, target_shape, scales, axes, attrs);
    interpolate->set_friendly_name(name_interpolate);

    // Pattern part 3: transpose
    auto order = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose = std::make_shared<ov::op::v1::Transpose>(interpolate, order);
    transpose->set_friendly_name(name_transpose);

    // Pattern part 4: sin
    auto sin = std::make_shared<ov::op::v0::Sin>(transpose);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

// This model has a similar structure but doesn't match the specific pattern.
static std::shared_ptr<ov::Model> create_model_not_to_match() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    auto mul_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 32}, {1.0f});
    auto multiply = std::make_shared<ov::op::v1::Multiply>(input, mul_const);
    multiply->set_friendly_name(name_multiply);

    auto sin = std::make_shared<ov::op::v0::Sin>(multiply);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

static void run_test(std::shared_ptr<ov::Model> model,
                     const std::unordered_map<std::string, bool>& expected_fp16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableFP16ComSinGenPatternForHiFiGAN>();

    precisions_map fp_convert_precision_map = {
        {ov::element::f32, ov::element::f16}
    };
    manager.register_pass<ov::pass::ConvertPrecision>(fp_convert_precision_map);

    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_fp16_disabled_status.find(op->get_friendly_name());
        if (it != expected_fp16_disabled_status.end()) {
            bool expected_status = it->second;
            if (expected_status) {
                ASSERT_TRUE(ov::fp16_compression_is_disabled(op))
                    << "FP16 compression is not disabled for node: " << op->get_friendly_name();
            } else {
                ASSERT_FALSE(ov::fp16_compression_is_disabled(op))
                    << "FP16 compression is unexpectedly disabled for node: " << op->get_friendly_name();
            }
        }
    }
}

TEST(TransformationTests, DisableFP16CompSinGen_Positive) {
    auto model = create_model_to_match();
    // In the matching pattern, sin should have FP16 compression disabled.
    std::unordered_map<std::string, bool> expected_status = {
        {name_multiply, true},
        {name_interpolate, true},
        {name_transpose, true},
        {name_sin, true}
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableFP16CompSinGen_Negative) {
    auto model = create_model_not_to_match();
    // In the non-matching model, no node should have FP16 compression disabled by the pass.
    std::unordered_map<std::string, bool> expected_status = {
        {name_multiply, false},
        {name_sin, false}
    };
    run_test(model, expected_status);
}
