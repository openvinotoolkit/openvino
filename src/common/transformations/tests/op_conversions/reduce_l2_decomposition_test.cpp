// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset4.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/op_conversions/reduce_l2_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, ReduceL2DecompositionTest) {
    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto axes = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i32, ngraph::Shape{1});
        auto reduce_l2 = std::make_shared<ngraph::opset4::ReduceL2>(data, axes, true);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{reduce_l2}, ngraph::ParameterVector{data, axes});
        manager.register_pass<ngraph::pass::ReduceL2Decomposition>();
    }

    {
        auto data = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic(1));
        auto axes = std::make_shared<ngraph::opset4::Parameter>(ngraph::element::i32, ngraph::Shape{1});
        auto pow = std::make_shared<ngraph::opset4::Power>(data, ngraph::opset4::Constant::create(ngraph::element::f32, ngraph::Shape{}, {2.0}));
        auto reduce_sum = std::make_shared<ngraph::opset4::ReduceSum>(pow, axes, true);
        auto sqrt = std::make_shared<ngraph::opset4::Sqrt>(reduce_sum);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{sqrt}, ngraph::ParameterVector{data, axes});
    }
}
