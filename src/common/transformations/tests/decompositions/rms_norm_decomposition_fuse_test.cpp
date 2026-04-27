// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//
// Tests that verify reference decompositions stored in
// `src/common/decompositions/` are always recognised and folded back into
// their corresponding internal fused op by the matching transformation.
//
// These tests act as a guard for both directions:
//   * decomposition authors must keep the produced sub-graph fusable;
//   * fusion authors must keep accepting the canonical decomposition shape.

#include <gtest/gtest.h>

#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/decompositions/rms_norm.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/pass/manager.hpp"
#include "ov_ops/rms.hpp"
#include "transformations/common_optimizations/rms_fusion.hpp"

using namespace testing;
using namespace ov::pass;

TEST_F(TransformationTestsF, DecompositionRmsNorm_FusedByRMSFusion_WithGamma) {
    const ov::Shape input_shape{1, 2, 6};
    const std::vector<float> gamma_values{0.029f, 0.014f, 0.003f, 0.013f, 0.015f, 0.009f};
    const float eps_value = 1e-5f;

    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        auto axes = ov::op::v0::Constant::create(ov::element::i64, ov::Shape{1}, {-1});
        auto eps = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{}, {eps_value});
        auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{6}, gamma_values);

        ov::pass::NodeRegistry reg;
        auto rms = ov::decompositions::rms_norm(reg, input, axes, eps, gamma);

        model = std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{input});
        manager.register_pass<RMSFusion>(/*force_tail_convert=*/false);
    }
    {
        auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, input_shape);
        auto gamma = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{6}, gamma_values);
        auto rms = std::make_shared<ov::op::internal::RMS>(input, gamma, eps_value, ov::element::f32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{rms}, ov::ParameterVector{input});
    }

    comparator.enable(FunctionsComparator::CmpValues::ACCURACY);
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    comparator.enable(FunctionsComparator::CmpValues::ATTRIBUTES);
}
