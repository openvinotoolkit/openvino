// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>

#include <openvino/core/model.hpp>
#include "openvino/opsets/opset1_decl.hpp"
#include <transformations/cpu_opset/common/pass/convert_to_leaky_relu.hpp>
#include <transformations/cpu_opset/common/op/leaky_relu.hpp>
#include <ov_ops/type_relaxed.hpp>
#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/op/prelu.hpp"

using namespace testing;
using namespace ov::intel_cpu;

class ConvertToLeakyReluTests : public TransformationTestsF {
protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ConvertToLeakyRelu>();
    }
};

TEST_F(ConvertToLeakyReluTests, StaticShape) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 16, 16});
        auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{}, {-2.f});
        auto prelu = std::make_shared<ov::opset1::PRelu>(input, slope);
        model = std::make_shared<ov::Model>(ov::OutputVector{prelu}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 16, 16});
        auto leaky_relu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ov::element::f32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{leaky_relu}, ov::ParameterVector{input});
    }
}

TEST_F(ConvertToLeakyReluTests, DynamicRank4) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{}, {-2.f});
        auto prelu = std::make_shared<ov::opset1::PRelu>(input, slope);
        model = std::make_shared<ov::Model>(ov::OutputVector{prelu}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic(4));
        auto leaky_relu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ov::element::f32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{leaky_relu}, ov::ParameterVector{input});
    }
}

TEST_F(ConvertToLeakyReluTests, FullyDynamic) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{}, {-2.f});
        auto prelu = std::make_shared<ov::opset1::PRelu>(input, slope);
        model = std::make_shared<ov::Model>(ov::OutputVector{prelu}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::PartialShape::dynamic());
        auto leaky_relu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ov::element::f32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{leaky_relu}, ov::ParameterVector{input});
    }
}

TEST_F(ConvertToLeakyReluTests, TypeRelaxed) {
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::u8, ov::Shape{1, 3, 16, 16});
        auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{}, {-2.f});
        auto relaxed_prelu = std::make_shared<ov::op::TypeRelaxed<ov::opset1::PRelu>>(
            ov::element::TypeVector{ov::element::f32, ov::element::f32},
            ov::element::TypeVector{ov::element::f32},
            ov::op::TemporaryReplaceOutputType(input, ov::element::f32).get(),
            ov::op::TemporaryReplaceOutputType(slope, ov::element::f32).get());
        model = std::make_shared<ov::Model>(ov::OutputVector{relaxed_prelu}, ov::ParameterVector{input});
    }
    {
        auto input = std::make_shared<ov::opset1::Parameter>(ov::element::u8, ov::Shape{1, 3, 16, 16});
        auto leaky_relu = std::make_shared<ov::intel_cpu::LeakyReluNode>(input, -2.f, ov::element::f32);
        model_ref = std::make_shared<ov::Model>(ov::OutputVector{leaky_relu}, ov::ParameterVector{input});
    }
}

TEST_F(ConvertToLeakyReluTests, Negative_NonScalarSlope) {
    auto input = std::make_shared<ov::opset1::Parameter>(ov::element::f32, ov::Shape{1, 3, 16, 16});
    auto slope = ov::opset1::Constant::create(ov::element::f32, ov::Shape{3}, {-2.f, -1.f, -2.f});
    auto prelu = std::make_shared<ov::opset1::PRelu>(input, slope);
    model = std::make_shared<ov::Model>(ov::OutputVector{prelu}, ov::ParameterVector{input});
    // model_ref intentionally omitted — transformation should not fire
}
