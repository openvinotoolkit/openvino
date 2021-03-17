// Copyright (C) 2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <queue>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <transformations/common_optimizations/clamp_fusion.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ngraph/pass/constant_folding.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"


using namespace testing;
using namespace ngraph;


TEST(TransformationTests, ClampFusion) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_const = opset5::Constant::create(element::f32, Shape{1}, {0.1});
        auto max_const = opset5::Constant::create(element::f32, Shape{1}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_const);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);
        f = std::make_shared<Function>(NodeVector{min}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ClampFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto clamp = std::make_shared<opset5::Clamp>(data, 0.1, 5);
        f_ref = std::make_shared<Function>(NodeVector{clamp}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ClampFusionScalars) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_const = opset5::Constant::create(element::f32, Shape{}, {0.1});
        auto max_const = opset5::Constant::create(element::f32, Shape{}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_const);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);
        f = std::make_shared<Function>(NodeVector{min}, ParameterVector{data});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ClampFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset1::Parameter>(element::f32, Shape{2, 2});
        auto clamp = std::make_shared<opset5::Clamp>(data, 0.1, 5);
        f_ref = std::make_shared<Function>(NodeVector{clamp}, ParameterVector{data});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(TransformationTests, ClampFusionNonConstMin) {
    std::shared_ptr<Function> f(nullptr), f_ref(nullptr);
    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_val = std::make_shared<opset5::Parameter>(element::f32, Shape{});
        auto max_const = opset5::Constant::create(element::f32, Shape{}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_val);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);
        f = std::make_shared<Function>(NodeVector{min}, ParameterVector{data, min_val});

        pass::Manager m;
        m.register_pass<pass::InitNodeInfo>();
        m.register_pass<pass::ClampFusion>();
        m.run_passes(f);
        ASSERT_NO_THROW(check_rt_info(f));
    }

    {
        auto data = std::make_shared<opset5::Parameter>(element::f32, Shape{2, 2});
        auto min_val = std::make_shared<opset5::Parameter>(element::f32, Shape{});
        auto max_const = opset5::Constant::create(element::f32, Shape{}, {5});
        auto max = std::make_shared<opset5::Maximum>(data, min_val);
        auto min = std::make_shared<opset5::Minimum>(max, max_const);
        f_ref = std::make_shared<Function>(NodeVector{min}, ParameterVector{data, min_val});
    }

    auto res = compare_functions(f, f_ref);
    ASSERT_TRUE(res.first) << res.second;
}
