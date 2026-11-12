// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <limits>
#include <memory>

#include "common_test_utils/ov_test_utils.hpp"
#include "intel_gpu/op/swoosh.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/add.hpp"
#include "openvino/op/constant.hpp"
#include "openvino/op/equal.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/parameter.hpp"
#include "openvino/op/result.hpp"
#include "openvino/op/select.hpp"
#include "openvino/op/softplus.hpp"
#include "openvino/op/subtract.hpp"
#include "openvino/pass/manager.hpp"
#include "plugin/transformations/swoosh_fusion.hpp"

namespace {

namespace swoosh_constants = ov::intel_gpu::op::swoosh_constants;

struct SwooshDecomposition {
    std::shared_ptr<ov::op::v0::Parameter> input;
    std::shared_ptr<ov::op::v4::SoftPlus> softplus;
    std::shared_ptr<ov::Model> model;
};

struct SwooshBuildOptions {
    bool reverse_commutative_inputs = false;
    bool use_different_scale_input = false;
    bool use_different_select_shifted = false;
    bool use_different_select_softplus = false;
};

SwooshDecomposition build_swoosh(float shift,
                                 float bias,
                                 bool expose_softplus = false,
                                 bool canonical_body = false,
                                 const ov::element::Type& data_type = ov::element::f32,
                                 const ov::PartialShape& input_shape = {-1, 128},
                                 const ov::Shape& constant_shape = {},
                                 bool use_f16_rounded_constants = false,
                                 const SwooshBuildOptions& options = {}) {
    SwooshDecomposition decomposition;
    decomposition.input = std::make_shared<ov::op::v0::Parameter>(data_type, input_shape);
    ov::ParameterVector parameters{decomposition.input};
    const auto constant_value = [&](float value) {
        return use_f16_rounded_constants ? static_cast<float>(ov::float16(value)) : value;
    };
    auto shift_constant = ov::op::v0::Constant::create(data_type, constant_shape, {constant_value(shift)});
    auto shifted = options.reverse_commutative_inputs
                       ? std::make_shared<ov::op::v1::Add>(shift_constant, decomposition.input)
                       : std::make_shared<ov::op::v1::Add>(decomposition.input, shift_constant);
    decomposition.softplus = std::make_shared<ov::op::v4::SoftPlus>(shifted);
    const auto overflow_limit = data_type == ov::element::f16
                                    ? static_cast<float>(std::numeric_limits<ov::float16>::max())
                                    : std::numeric_limits<float>::infinity();
    auto infinity = ov::op::v0::Constant::create(data_type,
                                                 constant_shape,
                                                 {overflow_limit});
    auto equal = options.reverse_commutative_inputs
                     ? std::make_shared<ov::op::v1::Equal>(infinity, decomposition.softplus)
                     : std::make_shared<ov::op::v1::Equal>(decomposition.softplus, infinity);
    std::shared_ptr<ov::Node> select_shifted = shifted;
    if (options.use_different_select_shifted) {
        auto other_shift = ov::op::v0::Constant::create(data_type, constant_shape, {constant_value(shift)});
        select_shifted = std::make_shared<ov::op::v1::Add>(decomposition.input, other_shift);
    }
    std::shared_ptr<ov::Node> select_softplus = decomposition.softplus;
    if (options.use_different_select_softplus) {
        select_softplus = std::make_shared<ov::op::v4::SoftPlus>(shifted);
    }
    auto select = std::make_shared<ov::op::v1::Select>(equal, select_shifted, select_softplus);
    auto slope = ov::op::v0::Constant::create(data_type, constant_shape, {constant_value(canonical_body ? -swoosh_constants::slope : swoosh_constants::slope)});
    std::shared_ptr<ov::op::v0::Parameter> scale_input = decomposition.input;
    if (options.use_different_scale_input) {
        scale_input = std::make_shared<ov::op::v0::Parameter>(data_type, input_shape);
        parameters.push_back(scale_input);
    }
    auto scaled = options.reverse_commutative_inputs
                      ? std::make_shared<ov::op::v1::Multiply>(scale_input, slope)
                      : std::make_shared<ov::op::v1::Multiply>(slope, scale_input);
    std::shared_ptr<ov::Node> body;
    if (canonical_body) {
        body = options.reverse_commutative_inputs ? std::make_shared<ov::op::v1::Add>(scaled, select)
                                                  : std::make_shared<ov::op::v1::Add>(select, scaled);
    } else {
        body = std::make_shared<ov::op::v1::Subtract>(select, scaled);
    }
    auto bias_constant = ov::op::v0::Constant::create(data_type, constant_shape, {constant_value(bias)});
    auto root = options.reverse_commutative_inputs ? std::make_shared<ov::op::v1::Add>(body, bias_constant)
                                                   : std::make_shared<ov::op::v1::Add>(bias_constant, body);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(root)};
    if (expose_softplus) {
        results.push_back(std::make_shared<ov::op::v0::Result>(decomposition.softplus));
    }
    decomposition.model = std::make_shared<ov::Model>(results, parameters);
    return decomposition;
}

std::shared_ptr<ov::intel_gpu::op::Swoosh> run_fusion(const std::shared_ptr<ov::Model>& model) {
    ov::pass::Manager manager;
    manager.register_pass<ov::intel_gpu::SwooshFusion>();
    manager.run_passes(model);
    return ov::as_type_ptr<ov::intel_gpu::op::Swoosh>(model->get_results()[0]->get_input_node_shared_ptr(0));
}

}  // namespace

TEST(SwooshFusionTest, FusesSwooshL) {
    auto decomposition = build_swoosh(-swoosh_constants::swoosh_l_offset, -swoosh_constants::swoosh_l_bias);
    auto swoosh = run_fusion(decomposition.model);

    ASSERT_NE(swoosh, nullptr);
    EXPECT_FLOAT_EQ(swoosh->get_offset(), swoosh_constants::swoosh_l_offset);
    EXPECT_FLOAT_EQ(swoosh->get_bias(), swoosh_constants::swoosh_l_bias);
    EXPECT_EQ(swoosh->get_input_node_shared_ptr(0), decomposition.input);
}

TEST(SwooshFusionTest, FusesSwooshR) {
    auto decomposition = build_swoosh(-swoosh_constants::swoosh_r_offset, -swoosh_constants::swoosh_r_bias);
    auto swoosh = run_fusion(decomposition.model);

    ASSERT_NE(swoosh, nullptr);
    EXPECT_FLOAT_EQ(swoosh->get_offset(), swoosh_constants::swoosh_r_offset);
    EXPECT_FLOAT_EQ(swoosh->get_bias(), swoosh_constants::swoosh_r_bias);
}

TEST(SwooshFusionTest, FusesCanonicalizedAddBody) {
    auto decomposition = build_swoosh(-swoosh_constants::swoosh_l_offset, -swoosh_constants::swoosh_l_bias, false, true);
    EXPECT_NE(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, FusesReversedCommutativeInputs) {
    SwooshBuildOptions options;
    options.reverse_commutative_inputs = true;
    auto decomposition =
        build_swoosh(-swoosh_constants::swoosh_l_offset, -swoosh_constants::swoosh_l_bias, false, true, ov::element::f32, {-1, 128}, {}, false, options);
    EXPECT_NE(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsDifferentScaleInput) {
    SwooshBuildOptions options;
    options.use_different_scale_input = true;
    auto decomposition =
        build_swoosh(-swoosh_constants::swoosh_l_offset, -swoosh_constants::swoosh_l_bias, false, false, ov::element::f32, {-1, 128}, {}, false, options);
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsDifferentSelectShiftedInput) {
    SwooshBuildOptions options;
    options.use_different_select_shifted = true;
    auto decomposition =
        build_swoosh(-swoosh_constants::swoosh_l_offset, -swoosh_constants::swoosh_l_bias, false, false, ov::element::f32, {-1, 128}, {}, false, options);
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsDifferentSelectSoftplusInput) {
    SwooshBuildOptions options;
    options.use_different_select_softplus = true;
    auto decomposition =
        build_swoosh(-swoosh_constants::swoosh_l_offset, -swoosh_constants::swoosh_l_bias, false, false, ov::element::f32, {-1, 128}, {}, false, options);
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsUnsupportedConstants) {
    auto decomposition = build_swoosh(-swoosh_constants::swoosh_l_offset, -0.04f);
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsExternallyVisibleIntermediate) {
    auto decomposition = build_swoosh(-swoosh_constants::swoosh_l_offset, -swoosh_constants::swoosh_l_bias, true);
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsEmptyConstantWithoutCrashing) {
    auto input = std::make_shared<ov::op::v0::Parameter>(ov::element::f32, ov::PartialShape{0});
    auto empty = ov::op::v0::Constant::create(ov::element::f32, ov::Shape{0}, std::vector<float>{});
    auto add = std::make_shared<ov::op::v1::Add>(input, empty);
    auto model = std::make_shared<ov::Model>(ov::OutputVector{add}, ov::ParameterVector{input});

    EXPECT_EQ(run_fusion(model), nullptr);
}

TEST(SwooshFusionTest, RejectsShapeExpandingSplatConstants) {
    auto decomposition =
        build_swoosh(-swoosh_constants::swoosh_l_offset, -swoosh_constants::swoosh_l_bias, false, false, ov::element::f32, ov::PartialShape{}, ov::Shape{2});
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, FusesZipformerShapedSplatConstants) {
    auto decomposition = build_swoosh(-swoosh_constants::swoosh_l_offset,
                                      -swoosh_constants::swoosh_l_bias,
                                      false,
                                      false,
                                      ov::element::f32,
                                      ov::PartialShape{4, -1, 384},
                                      ov::Shape{1, 1, 1});
    EXPECT_NE(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsF16RoundedConstantsInF32Graph) {
    auto decomposition = build_swoosh(-swoosh_constants::swoosh_l_offset,
                                      -swoosh_constants::swoosh_l_bias,
                                      false,
                                      false,
                                      ov::element::f32,
                                      ov::PartialShape{-1, 128},
                                      ov::Shape{},
                                      true);
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, FusesF16ConstantsInF16Graph) {
    auto decomposition = build_swoosh(-swoosh_constants::swoosh_l_offset, -swoosh_constants::swoosh_l_bias, false, false, ov::element::f16);
    EXPECT_NE(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshConstantsTest, PreservesOriginalValues) {
    EXPECT_EQ(swoosh_constants::slope, 0.08f);
    EXPECT_EQ(swoosh_constants::swoosh_l_offset, 4.0f);
    EXPECT_EQ(swoosh_constants::swoosh_l_bias, 0.035f);
    EXPECT_EQ(swoosh_constants::swoosh_r_offset, 1.0f);
    EXPECT_EQ(swoosh_constants::swoosh_r_bias, 0.3132616877555847f);
}
