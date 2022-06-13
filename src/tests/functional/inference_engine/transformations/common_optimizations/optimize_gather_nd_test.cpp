// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/optimize_gather_nd.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;

TEST_F(TransformationTestsF, OptimizeGatherND) {
    {
        // prepare inputs
        auto indices = ngraph::opset8::Constant::create(ngraph::element::i64, ngraph::Shape{2, 2}, {1, 0, 0, 1});
        auto data = std::make_shared<ngraph::opset8::Parameter>(ngraph::element::f32, ngraph::Shape{2, 2});

        // create GatherND operator
        auto gather_nd = std::make_shared<ngraph::opset8::GatherND>(data, indices);

        // create a graph
        function = std::make_shared<ngraph::Function>(ngraph::NodeVector{gather_nd}, ngraph::ParameterVector{data});

        manager.register_pass<ov::pass::OptimizerGatherND>();
    }
}
