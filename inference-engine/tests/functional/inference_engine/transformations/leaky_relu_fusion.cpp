// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset8.hpp>
#include <transformations/common_optimizations/leaky_relu_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;

TEST(TransformationTests, LeakyReluFusionConstant) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{1}, {0.1});
        auto multiply = std::make_shared<opset8::Multiply>(data, alpha);
        auto max = std::make_shared<opset8::Maximum>(data, multiply);
        f = std::make_shared<Function>(NodeVector{max}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::LeakyReluFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{1}, {0.1});
        auto leaky_relu = std::make_shared<opset8::PRelu>(data, alpha);
        f_ref = std::make_shared<Function>(NodeVector{leaky_relu}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LeakyReluFusionScalar) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{}, {0.1});
        auto multiply = std::make_shared<opset8::Multiply>(data, alpha);
        auto max = std::make_shared<opset8::Maximum>(data, multiply);
        f = std::make_shared<Function>(NodeVector{max}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::LeakyReluFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto alpha = opset8::Constant::create(element::f32, Shape{}, {0.1});
        auto leaky_relu = std::make_shared<opset8::PRelu>(data, alpha);
        f_ref = std::make_shared<Function>(NodeVector{leaky_relu}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, LeakyReluFusionParameter) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset8::Parameter>(element::f32, Shape{2, 2});
        auto alpha = std::make_shared<opset8::Parameter>(element::f32, Shape{});
        auto multiply = std::make_shared<opset8::Multiply>(data, alpha);
        auto max = std::make_shared<opset8::Maximum>(data, multiply);
        f = std::make_shared<Function>(NodeVector{max}, ParameterVector{data, alpha});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::LeakyReluFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto alpha = std::make_shared<opset8::Parameter>(element::f32, Shape{});
        auto leaky_relu = std::make_shared<opset8::PRelu>(data, alpha);
        f_ref = std::make_shared<Function>(NodeVector{leaky_relu}, ParameterVector{data, alpha});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
