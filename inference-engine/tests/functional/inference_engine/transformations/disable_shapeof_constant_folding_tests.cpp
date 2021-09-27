// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset6.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include <disable_shapeof_constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;

TEST(TransformationTests, DisableShapeOfConstantFolding) {
    std::shared_ptr<Function> f, f_ref;
    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto abs = std::make_shared<opset6::Abs>(shape_of);
        auto reshape = std::make_shared<opset6::Reshape>(data, abs, false);
        f = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::DisableShapeOfConstantFolding>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::f32, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto abs = std::make_shared<opset6::Abs>(shape_of);
        auto reshape = std::make_shared<opset6::Reshape>(data, abs, false);
        f_ref = std::make_shared<Function>(NodeVector{reshape}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ShapeOfShapeOfConstantFolding) {
    std::shared_ptr<Function> f, f_ref;
    {
        auto data = std::make_shared<opset6::Parameter>(element::i64, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto reshape = std::make_shared<opset6::Reshape>(data, shape_of, false);
        auto rank = std::make_shared<opset6::ShapeOf>(shape_of);
        auto mul = std::make_shared<opset6::Multiply>(reshape, rank);
        f = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::DisableShapeOfConstantFolding>();
        m.register_pass<pass::ConstantFolding>();
        m.run_passes(f);
    }

    {
        auto data = std::make_shared<opset6::Parameter>(element::i64, Shape{1, 4, 10, 10});
        auto shape_of = std::make_shared<opset6::ShapeOf>(data);
        auto reshape = std::make_shared<opset6::Reshape>(data, shape_of, false);
        auto mul = std::make_shared<opset6::Multiply>(reshape, opset6::Constant::create(element::i64, Shape{1}, {4}));
        f_ref = std::make_shared<Function>(NodeVector{mul}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
