// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/op_conversions/convert_divide.hpp>
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
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {1.5});
        auto pow = std::make_shared<ngraph::opset1::Power>(divide_constant,
                                                           ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1}));
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data, pow);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data});
    }
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
}

TEST_F(TransformationTestsF, ConvertDivideScalar) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.5});
        auto divide = std::make_shared<ngraph::opset1::Divide>(data, divide_constant);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{divide}, ngraph::ParameterVector{data});

        NGRAPH_CHECK(divide->get_output_partial_shape(0).rank().get_length() == 0);

        manager.register_pass<ngraph::pass::ConvertDivide>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{});
        auto divide_constant = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {1.5});
        auto pow = std::make_shared<ngraph::opset1::Power>(divide_constant,
                                                           ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{}, {-1}));
        auto mul = std::make_shared<ngraph::opset1::Multiply>(data, pow);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{mul}, ngraph::ParameterVector{data});

        NGRAPH_CHECK(mul->get_output_partial_shape(0).rank().get_length() == 0);
    }
}
