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

struct SwooshDecomposition {
    std::shared_ptr<ov::op::v0::Parameter> input;
    std::shared_ptr<ov::op::v4::SoftPlus> softplus;
    std::shared_ptr<ov::Model> model;
};

SwooshDecomposition build_swoosh(float shift,
                                 float bias,
                                 bool expose_softplus = false,
                                 bool canonical_body = false,
                                 const ov::element::Type& data_type = ov::element::f32,
                                 const ov::PartialShape& input_shape = {-1, 128},
                                 const ov::Shape& constant_shape = {},
                                 bool use_f16_rounded_constants = false) {
    SwooshDecomposition decomposition;
    decomposition.input = std::make_shared<ov::op::v0::Parameter>(data_type, input_shape);
    const auto constant_value = [&](float value) {
        return use_f16_rounded_constants ? static_cast<float>(ov::float16(value)) : value;
    };
    auto shift_constant = ov::op::v0::Constant::create(data_type, constant_shape, {constant_value(shift)});
    auto shifted = std::make_shared<ov::op::v1::Add>(decomposition.input, shift_constant);
    decomposition.softplus = std::make_shared<ov::op::v4::SoftPlus>(shifted);
    const auto overflow_limit = data_type == ov::element::f16
                                    ? static_cast<float>(std::numeric_limits<ov::float16>::max())
                                    : std::numeric_limits<float>::infinity();
    auto infinity = ov::op::v0::Constant::create(data_type,
                                                 constant_shape,
                                                 {overflow_limit});
    auto equal = std::make_shared<ov::op::v1::Equal>(decomposition.softplus, infinity);
    auto select = std::make_shared<ov::op::v1::Select>(equal, shifted, decomposition.softplus);
    auto slope = ov::op::v0::Constant::create(data_type,
                                              constant_shape,
                                              {constant_value(canonical_body ? -0.08f : 0.08f)});
    auto scaled = std::make_shared<ov::op::v1::Multiply>(slope, decomposition.input);
    std::shared_ptr<ov::Node> body;
    if (canonical_body) {
        body = std::make_shared<ov::op::v1::Add>(select, scaled);
    } else {
        body = std::make_shared<ov::op::v1::Subtract>(select, scaled);
    }
    auto bias_constant = ov::op::v0::Constant::create(data_type, constant_shape, {constant_value(bias)});
    auto root = std::make_shared<ov::op::v1::Add>(bias_constant, body);

    ov::ResultVector results{std::make_shared<ov::op::v0::Result>(root)};
    if (expose_softplus) {
        results.push_back(std::make_shared<ov::op::v0::Result>(decomposition.softplus));
    }
    decomposition.model = std::make_shared<ov::Model>(results, ov::ParameterVector{decomposition.input});
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
    auto decomposition = build_swoosh(-4.0f, -0.035f);
    auto swoosh = run_fusion(decomposition.model);

    ASSERT_NE(swoosh, nullptr);
    EXPECT_FLOAT_EQ(swoosh->get_offset(), 4.0f);
    EXPECT_FLOAT_EQ(swoosh->get_bias(), 0.035f);
    EXPECT_EQ(swoosh->get_input_node_shared_ptr(0), decomposition.input);
}

TEST(SwooshFusionTest, FusesSwooshR) {
    auto decomposition = build_swoosh(-1.0f, -0.3132616877555847f);
    auto swoosh = run_fusion(decomposition.model);

    ASSERT_NE(swoosh, nullptr);
    EXPECT_FLOAT_EQ(swoosh->get_offset(), 1.0f);
    EXPECT_FLOAT_EQ(swoosh->get_bias(), 0.3132616877555847f);
}

TEST(SwooshFusionTest, FusesCanonicalizedAddBody) {
    auto decomposition = build_swoosh(-4.0f, -0.035f, false, true);
    EXPECT_NE(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsUnsupportedConstants) {
    auto decomposition = build_swoosh(-4.0f, -0.04f);
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsExternallyVisibleIntermediate) {
    auto decomposition = build_swoosh(-4.0f, -0.035f, true);
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
    auto decomposition = build_swoosh(-4.0f,
                                      -0.035f,
                                      false,
                                      false,
                                      ov::element::f32,
                                      ov::PartialShape{},
                                      ov::Shape{2});
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, FusesZipformerShapedSplatConstants) {
    auto decomposition = build_swoosh(-4.0f,
                                      -0.035f,
                                      false,
                                      false,
                                      ov::element::f32,
                                      ov::PartialShape{4, -1, 384},
                                      ov::Shape{1, 1, 1});
    EXPECT_NE(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, RejectsF16RoundedConstantsInF32Graph) {
    auto decomposition = build_swoosh(-4.0f,
                                      -0.035f,
                                      false,
                                      false,
                                      ov::element::f32,
                                      ov::PartialShape{-1, 128},
                                      ov::Shape{},
                                      true);
    EXPECT_EQ(run_fusion(decomposition.model), nullptr);
}

TEST(SwooshFusionTest, FusesF16ConstantsInF16Graph) {
    auto decomposition = build_swoosh(-4.0f,
                                      -0.035f,
                                      false,
                                      false,
                                      ov::element::f16);
    EXPECT_NE(run_fusion(decomposition.model), nullptr);
}