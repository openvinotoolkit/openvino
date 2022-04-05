// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>


#include <ngraph/function.hpp>
#include <ngraph/opsets/opset2.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <transformations/op_conversions/gelu7_downgrade.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, Gelu7Downgrade) {
    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto gelu = std::make_shared<ngraph::opset7::Gelu>(input, ngraph::op::GeluApproximationMode::ERF);

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gelu}, ngraph::ParameterVector{input});

        manager.register_pass<ngraph::pass::Gelu7Downgrade>();
    }

    {
        auto input = std::make_shared<ngraph::opset7::Parameter>(ngraph::element::f32, ngraph::Shape{1, 2, 3});
        auto gelu = std::make_shared<ngraph::opset2::Gelu>(input);

        function_ref = std::make_shared<ngraph::Function>(ngraph::NodeVector{gelu}, ngraph::ParameterVector{input});
    }
}

