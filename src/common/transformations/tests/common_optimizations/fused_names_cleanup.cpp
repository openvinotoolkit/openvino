// Copyright (C) 2018-2022 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#define _USE_MATH_DEFINES

#include <gtest/gtest.h>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset9.hpp>
#include <ngraph/pass/constant_folding.hpp>
#include <ngraph/pass/manager.hpp>
#include <transformations/common_optimizations/fused_names_cleanup.hpp>
#include <transformations/init_node_info.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ngraph;

TEST(TransformationTests, FusedNamesCleanup) {
    std::shared_ptr<ngraph::Function> function(nullptr), function_ref(nullptr);
    {
        auto data =
            std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});

        auto add1_const =
            opset9::Constant::create(element::f32, Shape{1}, {1.0});
        auto add2_const =
            opset9::Constant::create(element::f32, Shape{1}, {2.0});
        auto add1 = std::make_shared<opset9::Add>(add1_const, add2_const);
        auto add2 = std::make_shared<opset9::Add>(data, add1);
        function = std::make_shared<Function>(NodeVector{add2}, ParameterVector{data});

        ngraph::pass::Manager manager;
        manager.register_pass<pass::InitNodeInfo>();
        manager.register_pass<pass::ConstantFolding>();
        manager.run_passes(function);
        ASSERT_NO_THROW(check_rt_info(function));

        manager.register_pass<ov::pass::FusedNamesCleanup>();
        manager.run_passes(function);
        ASSERT_THROW(check_rt_info(function), ngraph::ngraph_error);
    }
    {
        auto data =
            std::make_shared<opset9::Parameter>(element::f32, Shape{2, 2});

        auto add_const =
            opset9::Constant::create(element::f32, Shape{1}, {3.0});
        auto add = std::make_shared<opset9::Add>(data, add_const);
        function_ref = std::make_shared<Function>(NodeVector{add}, ParameterVector{data});
    }
    const FunctionsComparator func_comparator = FunctionsComparator::with_default().enable(FunctionsComparator::RUNTIME_KEYS);
    const FunctionsComparator::Result result = func_comparator(function, function_ref);
    ASSERT_TRUE(result.valid);
}
