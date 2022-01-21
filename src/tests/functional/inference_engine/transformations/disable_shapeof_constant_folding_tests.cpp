// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <memory>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <queue>
#include <string>
#include <transformations/common_optimizations/disable_shapeof_constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST_F(TransformationTestsF, DisableShapeOfConstantFolding) {
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto abs = std::make_shared<opset6::Abs>(shape_of);
        auto reshape = std::make_shared<opset6::Reshape>(data, abs, false);
        function = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});

        manager.register_pass<pass::DisableShapeOfConstantFolding>();
        manager.register_pass<pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto abs = std::make_shared<opset6::Abs>(shape_of);
        auto reshape = std::make_shared<opset6::Reshape>(data, abs, false);
        function_ref = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
    }
}

TEST_F(TransformationTestsF, ShapeOfShapeOfConstantFolding) {
    std::shared_ptr<Function> f, f_ref;
    {
        auto data = std::make_shared<opset6::Parameter>(element::i64, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto reshape = std::make_shared<opset6::Reshape>(data, shape_of, false);
        auto rank = std::make_shared<opset6::ShapeOf>(shape_of);
        auto mul = std::make_shared<opset6::Multiply>(reshape, rank);
        function = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        manager.register_pass<pass::DisableShapeOfConstantFolding>();
        manager.register_pass<pass::ConstantFolding>();
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::i64, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto reshape = std::make_shared<opset6::Reshape>(data, shape_of, false);
        auto mul = std::make_shared<opset6::Multiply>(reshape, opset6::Constant::create(element::i64, Shape{1}, {4}));
        function_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});
    }
}
