// Copyright (C) 2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <transformations/op_conversions/convert_subtract.hpp>
#include <transformations/low_precision/mark_dequantization_subgraph.hpp>
#include <transformations/utils/utils.hpp>
#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, ConvertSubtract) {
    {
        auto data1 = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto constant = opset8::Constant::create(element::f32, Shape{1}, {1.5});
        auto sub1 = std::make_shared<opset8::Subtract>(data1, constant);
        auto data2 = std::make_shared<opset8::Parameter>(element::f32, Shape{2});
        auto sub2 = std::make_shared<opset8::Subtract>(data1, data2);

        function = std::make_shared<Function>(NodeVector{sub1, sub2}, ParameterVector{data1, data2});

        manager.register_pass<pass::ConvertSubtract>();
    }

    {
        auto data1 = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto constant = opset8::Constant::create(element::f32, Shape{1}, {-1.5});
        auto add1 = std::make_shared<opset8::Add>(data1, constant);
        auto data2 = std::make_shared<opset8::Parameter>(element::f32, Shape{2});
        auto neg = std::make_shared<opset8::Multiply>(data2, opset8::Constant::create(element::f32, Shape{}, {-1}));
        auto add2 = std::make_shared<opset8::Add>(data1, neg);

        function_ref = std::make_shared<Function>(NodeVector{add1, add2}, ParameterVector{data1, data2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    enable_accuracy_check();
}

TEST_F(TransformationTestsF, ConvertSubtractWithConstant) {
    {
        auto data1 = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto constant = opset8::Constant::create(element::f32, Shape{1}, {1.5});
        auto sub1 = std::make_shared<opset8::Subtract>(data1, constant);
        auto data2 = std::make_shared<opset8::Parameter>(element::f32, Shape{2});
        auto sub2 = std::make_shared<opset8::Subtract>(data1, data2);

        function = std::make_shared<Function>(NodeVector{sub1, sub2}, ParameterVector{data1, data2});

        manager.register_pass<pass::ConvertSubtractWithConstant>();
    }

    {
        auto data1 = std::make_shared<opset8::Parameter>(element::f32, Shape{3, 1, 2});
        auto constant = opset8::Constant::create(element::f32, Shape{1}, {-1.5});
        auto add = std::make_shared<opset8::Add>(data1, constant);
        auto data2 = std::make_shared<opset8::Parameter>(element::f32, Shape{2});
        auto sub = std::make_shared<opset8::Subtract>(data1, data2);

        function_ref = std::make_shared<Function>(NodeVector{add, sub}, ParameterVector{data1, data2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
    enable_accuracy_check();
}

TEST_F(TransformationTestsF, ConvertSubtractDequantizationSubgraph) {
    auto data = std::make_shared<opset8::Parameter>(element::u8, Shape{1, 3, 2, 2});
    auto convert = std::make_shared<opset8::Convert>(data, element::f32);
    auto zero_point = opset8::Constant::create(element::f32, Shape{1}, {2});
    auto sub = std::make_shared<opset8::Subtract>(convert, zero_point);
    auto scale = opset8::Constant::create(element::f32, Shape{1}, {3});
    auto mul = std::make_shared<opset8::Multiply>(sub, scale);

    function = std::make_shared<Function>(mul, ParameterVector{data});

    manager.register_pass<pass::MarkDequantizationSubgraph>(element::TypeVector{element::u8});
    manager.register_pass<pass::ConvertSubtract>();
}

TEST_F(TransformationTestsF, ConvertSubtractUnsignedType) {
    auto data = std::make_shared<opset8::Parameter>(element::u32, Shape{1, 3, 2, 2});
    auto constant = opset8::Constant::create(element::u32, Shape{1}, {2});
    auto sub = std::make_shared<opset8::Subtract>(data, constant);

    function = std::make_shared<Function>(sub, ParameterVector{data});

    manager.register_pass<pass::ConvertSubtract>();
}
