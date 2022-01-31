// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <transformations/op_conversions/batch_norm_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, BatchNormDecompositionDynamic) {
    {
        auto input = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, ngraph::PartialShape::dynamic());
        auto gamma = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {3});
        auto beta = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {3});
        auto mean = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {3});
        auto var = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{3}, {3});
        auto broadcast = std::make_shared<ngraph::opset1::BatchNormInference>(input, gamma, beta, mean, var, 0.001);
        broadcast->set_friendly_name("broadcast");

        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{broadcast}, ngraph::ParameterVector{input});
        function_ref = ngraph::clone_function(*function);
        manager.register_pass<ngraph::pass::BatchNormDecomposition>();
    }
}