// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
#include <transformations/common_optimizations/mark_precision_sensitive_divides.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertDivide) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::ConvertDivide>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1. / 1.5});
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data, divide_constant);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideInverse) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1});
        auto divide = std::make_shared<ngraph::opset1::Divide>(divide_constant, data);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::ConvertDivide>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{3, 1, 2});
        auto constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1.0});
        auto pow = std::make_shared<ngraph::opset1::Power>(data, constant);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{pow}, ngraph::ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}


TEST_F(TransformationTestsF, ConvertDivideNegative) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {2});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});

        manager.register_pass<ngraph::pass::ConvertDivide>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::i32, ngraph::Shape{3, 1, 2});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::i32, ngraph::Shape{1}, {2});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideScalar) {
    {
        auto data1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto data2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data1, data2);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data1, data2});

        NGRAPH_CHECK(divide->get_output_partial_shape(0).rank().get_length() == 0);

        manager.register_pass<ngraph::pass::ConvertDivide>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto pow_input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto pow = std::make_shared<ngraph::opset1::Power>(pow_input,
                                                           ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1}));
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data, pow);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data, pow_input});

        NGRAPH_CHECK(mul->get_output_partial_shape(0).rank().get_length() == 0);
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideWithConstantPositive) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.5});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::ConvertDivideWithConstant>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1. / 1.5});
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data, divide_constant);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideWithConstantNegative) {
    {
        auto data1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto data2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data1, data2);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data1, data2});
        manager.register_pass<ngraph::pass::ConvertDivideWithConstant>();
    }

    {
        auto data1 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto data2 = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data1, data2);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data1, data2});
    }
    comparator.enable(FunctionsComparator::CmpValues::CONST_VALUES);
}

TEST_F(TransformationTestsF, ConvertDivideFP16ShapeOfSubgraphNegative) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f16, ngraph::Shape{1, 3, 22, 22});
        auto gather = ngraph::op::util::node_to_get_shape_value_of_indices_from_shape_source(data, {2, 3});
        auto convert = std::make_shared<ngraph::opset1::Convert>(gather, ngraph::element::f16);
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f16, ngraph::Shape{1}, {0.5});
        auto divide = std::make_shared<ngraph::opset1::Divide>(convert, divide_constant);
        auto convert_after = std::make_shared<ngraph::opset1::Convert>(divide, ngraph::element::i32);

        ngraph::opset1::Interpolate::Attributes interp_attr;
        interp_attr.antialias = false;
        interp_attr.axes = {2, 3};
        interp_attr.mode = "nearest";
        interp_attr.pads_begin = {0, 0, 0, 0};
        interp_attr.pads_end = {0, 0, 0, 0};

        auto interpolate = std::make_shared<ngraph::opset1::Interpolate>(data, convert_after, interp_attr);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{interpolate}, ngraph::ParameterVector{data});

        ov::pass::MarkPrecisionSensitiveDivides().run_on_model(function);
        manager.register_pass<ngraph::pass::ConvertDivide>();
    }
}
