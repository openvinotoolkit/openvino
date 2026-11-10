// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/cpu_opset/common/pass/disable_bf16_comp_cumsum_sin_gen.hpp"

#include <gtest/gtest.h>

#include <memory>
#include <string>
#include <unordered_map>

#include "openvino/core/model.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/cum_sum.hpp"
#include "openvino/op/interpolate.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/sin.hpp"
#include "openvino/op/transpose.hpp"
#include "openvino/pass/manager.hpp"
#include "transformations/rt_info/disable_precision_conversion.hpp"

using namespace testing;
using namespace ov::intel_cpu;

namespace {

// Names used to look up matched nodes after the pass has run.
const std::string name_transpose_pre = "transpose_pre";
const std::string name_interp_pre = "interp_pre";
const std::string name_transpose_pre_cumsum = "transpose_pre_cumsum";
const std::string name_cumsum = "cumsum";
const std::string name_mul_after_cumsum = "mul_after_cumsum";
const std::string name_transpose_after_mul = "transpose_after_mul";
const std::string name_scale_mul = "scale_mul";
const std::string name_interp_after = "interp_after";
const std::string name_sin = "sin";

ov::op::util::InterpolateBase::InterpolateAttrs make_interp_attrs() {
    ov::op::util::InterpolateBase::InterpolateAttrs attrs;
    attrs.mode = ov::op::util::InterpolateBase::InterpolateMode::LINEAR;
    attrs.shape_calculation_mode = ov::op::util::InterpolateBase::ShapeCalcMode::SCALES;
    attrs.nearest_mode = ov::op::util::InterpolateBase::NearestMode::ROUND_PREFER_FLOOR;
    return attrs;
}

//   Transpose -> Interpolate -> Transpose -> CumSum
//     -> Multiply -> Multiply -> Transpose -> Interpolate -> Sin.
std::shared_ptr<ov::Model> create_full_chain_model() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    auto order_pre = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose_pre = std::make_shared<ov::op::v1::Transpose>(input, order_pre);
    transpose_pre->set_friendly_name(name_transpose_pre);

    auto sizes_pre = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {64L, 1L});
    auto scales_pre = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 1.0f});
    auto interp_pre =
        std::make_shared<ov::op::v4::Interpolate>(transpose_pre, sizes_pre, scales_pre, make_interp_attrs());
    interp_pre->set_friendly_name(name_interp_pre);

    auto order_pre_cumsum = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose_pre_cumsum = std::make_shared<ov::op::v1::Transpose>(interp_pre, order_pre_cumsum);
    transpose_pre_cumsum->set_friendly_name(name_transpose_pre_cumsum);

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto cumsum = std::make_shared<ov::op::v0::CumSum>(transpose_pre_cumsum, axis);
    cumsum->set_friendly_name(name_cumsum);

    auto mul_after_cumsum_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 64}, {6.2832f});
    auto mul_after_cumsum = std::make_shared<ov::op::v1::Multiply>(cumsum, mul_after_cumsum_const);
    mul_after_cumsum->set_friendly_name(name_mul_after_cumsum);

    auto scale_mul_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 1}, {1.0f});
    auto scale_mul = std::make_shared<ov::op::v1::Multiply>(mul_after_cumsum, scale_mul_const);
    scale_mul->set_friendly_name(name_scale_mul);

    auto order_after_mul = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose_after_mul = std::make_shared<ov::op::v1::Transpose>(scale_mul, order_after_mul);
    transpose_after_mul->set_friendly_name(name_transpose_after_mul);

    auto sizes_after = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {128L, 1L});
    auto scales_after = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 1.0f});
    auto interp_after =
        std::make_shared<ov::op::v4::Interpolate>(transpose_after_mul, sizes_after, scales_after, make_interp_attrs());
    interp_after->set_friendly_name(name_interp_after);

    auto sin = std::make_shared<ov::op::v0::Sin>(interp_after);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

// Same downstream chain but missing the second Multiply between the two
// Transposes — must not match.
std::shared_ptr<ov::Model> create_model_missing_scale_mul() {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{1, 32});

    auto order_pre = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose_pre = std::make_shared<ov::op::v1::Transpose>(input, order_pre);
    transpose_pre->set_friendly_name(name_transpose_pre);

    auto sizes_pre = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {64L, 1L});
    auto scales_pre = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 1.0f});
    auto interp_pre =
        std::make_shared<ov::op::v4::Interpolate>(transpose_pre, sizes_pre, scales_pre, make_interp_attrs());
    interp_pre->set_friendly_name(name_interp_pre);

    auto order_pre_cumsum = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose_pre_cumsum = std::make_shared<ov::op::v1::Transpose>(interp_pre, order_pre_cumsum);
    transpose_pre_cumsum->set_friendly_name(name_transpose_pre_cumsum);

    auto axis = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{}, {1});
    auto cumsum = std::make_shared<ov::op::v0::CumSum>(transpose_pre_cumsum, axis);
    cumsum->set_friendly_name(name_cumsum);

    auto mul_after_cumsum_const = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{1, 64}, {6.2832f});
    auto mul_after_cumsum = std::make_shared<ov::op::v1::Multiply>(cumsum, mul_after_cumsum_const);
    mul_after_cumsum->set_friendly_name(name_mul_after_cumsum);

    auto order_after_mul = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {1, 0});
    auto transpose_after_mul = std::make_shared<ov::op::v1::Transpose>(mul_after_cumsum, order_after_mul);
    transpose_after_mul->set_friendly_name(name_transpose_after_mul);

    // Missing scale Multiply — Interpolate is fed directly by the Transpose.
    auto sizes_after = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{2}, {64L, 1L});
    auto scales_after = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{2}, {2.0f, 1.0f});
    auto interp_after =
        std::make_shared<ov::op::v4::Interpolate>(transpose_after_mul, sizes_after, scales_after, make_interp_attrs());
    interp_after->set_friendly_name(name_interp_after);

    auto sin = std::make_shared<ov::op::v0::Sin>(interp_after);
    sin->set_friendly_name(name_sin);

    return std::make_shared<ov::Model>(ov::OutputVector{sin}, ov::ParameterVector{input});
}

void run_test(const std::shared_ptr<ov::Model>& model,
              const std::unordered_map<std::string, bool>& expected_bf16_disabled_status) {
    ov::pass::Manager manager;
    manager.register_pass<DisableBF16CompCumSumSinGen>();
    manager.run_passes(model);

    for (const auto& op : model->get_ops()) {
        auto it = expected_bf16_disabled_status.find(op->get_friendly_name());
        if (it == expected_bf16_disabled_status.end()) {
            continue;
        }
        if (it->second) {
            ASSERT_TRUE(ov::is_conversion_disabled(op, ov::element::f32, ov::element::bf16))
                << "BF16 conversion is not disabled for node: " << op->get_friendly_name();
        } else {
            ASSERT_FALSE(ov::is_conversion_disabled(op, ov::element::f32, ov::element::bf16))
                << "BF16 conversion is unexpectedly disabled for node: " << op->get_friendly_name();
        }
    }
}

}  // namespace

TEST(TransformationTests, DisableBF16CompCumSumSinGen_Positive) {
    auto model = create_full_chain_model();
    // Core matched nodes are marked as disabled for BF16 conversion.
    std::unordered_map<std::string, bool> expected_status = {
        {name_transpose_pre, true},
        {name_interp_pre, true},
        {name_transpose_pre_cumsum, true},
        {name_cumsum, true},
        {name_mul_after_cumsum, true},
        {name_scale_mul, true},
        {name_transpose_after_mul, true},
        {name_interp_after, true},
        {name_sin, true},
    };
    run_test(model, expected_status);
}

TEST(TransformationTests, DisableBF16CompCumSumSinGen_MissingScaleMultiply_NoOp) {
    auto model = create_model_missing_scale_mul();
    std::unordered_map<std::string, bool> expected_status = {
        {name_transpose_pre, false},
        {name_interp_pre, false},
        {name_transpose_pre_cumsum, false},
        {name_cumsum, false},
        {name_mul_after_cumsum, false},
        {name_transpose_after_mul, false},
        {name_interp_after, false},
        {name_sin, false},
    };
    run_test(model, expected_status);
}
