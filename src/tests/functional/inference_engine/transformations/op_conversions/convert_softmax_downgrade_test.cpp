// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/convert_softmax_downgrade.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertSoftMax8ToSoftMax1) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        int64_t axis = 1;
        auto softmax_8 = std::make_shared<ngraph::opset8::Softmax>(data, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{softmax_8}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::ConvertSoftMax8ToSoftMax1>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        size_t axis = 1;
        auto softmax_1 = std::make_shared<ngraph::opset1::Softmax>(data, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{softmax_1}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertSoftMax8ToSoftMax1_negative_axis) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        int64_t axis = -1;
        auto softmax_8 = std::make_shared<ngraph::opset8::Softmax>(data, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{softmax_8}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::ConvertSoftMax8ToSoftMax1>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{2, 3});
        size_t axis = 1;
        auto softmax_1 = std::make_shared<ngraph::opset1::Softmax>(data, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{softmax_1}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ConvertSoftMax8ToSoftMax1_input_rank_5) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5, 5, 5});
        int64_t axis = -2;
        auto softmax_8 = std::make_shared<ngraph::opset8::Softmax>(data, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{softmax_8}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::ConvertSoftMax8ToSoftMax1>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 3, 5, 5, 5});
        size_t axis = 3;
        auto softmax_1 = std::make_shared<ngraph::opset1::Softmax>(data, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{softmax_1}, ngraph::ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, negative_ConvertSoftMax8ToSoftMax1_dynamic_rank) {
    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        int64_t axis = -3;
        auto softmax_8 = std::make_shared<ngraph::opset8::Softmax>(data, axis);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{softmax_8}, ngraph::ParameterVector{data});
        manager.register_pass<ngraph::pass::ConvertSoftMax8ToSoftMax1>();
    }

    {
        auto data = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        int64_t axis = -3;
        auto softmax_8 = std::make_shared<ngraph::opset8::Softmax>(data, axis);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{softmax_8}, ngraph::ParameterVector{data});
    }
}
