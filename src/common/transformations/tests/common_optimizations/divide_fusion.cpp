// Copyright (C) 2018-2026 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "transformations/common_optimizations/divide_fusion.hpp"

#include <gtest/gtest.h>

#include "common_test_utils/ov_test_utils.hpp"
#include "openvino/core/model.hpp"
#include "openvino/op/divide.hpp"
#include "openvino/op/multiply.hpp"
#include "openvino/op/power.hpp"
#include "openvino/opsets/opset1_decl.hpp"

using namespace ov;

class DivideFusionTest : public TransformationTestsF {
public:
    DivideFusionTest() {
        comparator.enable(FunctionsComparator::CONST_VALUES);
    }

protected:
    void SetUp() override {
        TransformationTestsF::SetUp();
        manager.register_pass<ov::pass::DivideFusion>();
    }
};

TEST_F(DivideFusionTest, MultiplyByPowerNegativeOne) {
    {
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto pow_constant = opset1::Constant::create(element::f32, Shape{1}, {-1});
        auto pow = std::make_shared<opset1::Power>(data2, pow_constant);
        auto mul = std::make_shared<opset1::Multiply>(data1, pow);

        model = std::make_shared<ov::Model>(OutputVector{mul}, ParameterVector{data1, data2});
    }
    {
        auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
        auto divide = std::make_shared<opset1::Divide>(data1, data2);

        model_ref = std::make_shared<ov::Model>(OutputVector{divide}, ParameterVector{data1, data2});
    }
}

TEST_F(DivideFusionTest, MultiplyByPowerNotNegativeOne) {
    auto data1 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
    auto data2 = std::make_shared<opset1::Parameter>(element::f32, Shape{3, 1, 2});
    auto pow_constant = opset1::Constant::create(element::f32, Shape{1}, {-1.01});
    auto pow = std::make_shared<opset1::Power>(data2, pow_constant);
    auto mul = std::make_shared<opset1::Multiply>(data1, pow);

    model = std::make_shared<ov::Model>(OutputVector{mul}, ParameterVector{data1, data2});
}
