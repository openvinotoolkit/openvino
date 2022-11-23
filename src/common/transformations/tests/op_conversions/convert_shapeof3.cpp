// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <transformations/op_conversions/convert_shapeof3.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ConvertShapeOf3WithI64) {
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input, ngraph::element::i64);
        shapeof->set_friendly_name("shapeof");

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{shapeof}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::ConvertShapeOf3>();
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto shapeof = std::make_shared<ngraph::opset1::ShapeOf>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{shapeof}, ngraph::ParameterVector{input});
    }
}

TEST_F(TransformationTestsF, ConvertShapeOf3WithI32) {
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto shapeof = std::make_shared<ngraph::opset3::ShapeOf>(input, ngraph::element::i32);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{shapeof}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::ConvertShapeOf3>();
    }

    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto shapeof = std::make_shared<ngraph::opset1::ShapeOf>(input);
        auto convert = std::make_shared<ngraph::opset1::Convert>(shapeof, ngraph::element::i32);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{convert}, ngraph::ParameterVector{input});
    }
}
