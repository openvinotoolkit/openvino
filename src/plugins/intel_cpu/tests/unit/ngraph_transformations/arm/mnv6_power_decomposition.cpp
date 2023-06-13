// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <transformations/cpu_opset/arm/pass/mvn6_power_decomposition.hpp>
#include <transformations/init_node_info.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>
#include <utility>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

static std::shared_ptr<ov::Model> createInitGraph(const std::shared_ptr<ngraph::opset1::Parameter>& param, float power_value) {
    auto power_const = ngraph::opset1::Constant::create(ngraph::element::f32, ngraph::Shape{1}, {power_value});
    auto reduce = std::make_shared<ov::opset8::Power>(param, power_const);
    return std::make_shared<ngraph::Function>(ngraph::NodeVector{ reduce }, ngraph::ParameterVector{ param });
}

static std::shared_ptr<ov::Model> createRefGraph(const ov::Shape& param_shape) {
    auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, param_shape);
    auto multiply = std::make_shared<ngraph::opset1::Multiply>(param, param);
    return std::make_shared<ngraph::Function>(ngraph::NodeVector{ multiply }, ngraph::ParameterVector{ param });
}

static void registerAndRunPass(std::shared_ptr<ov::Model> function) {
    ov::pass::Manager manager;
    manager.register_pass<MVN6PowerDecomposition>();
    manager.run_passes(std::move(function));
}

static ngraph::Shape param_shape = ngraph::Shape{2, 19, 2, 9};

TEST(MVN6PowerDecompositionTest, CheckMVN6PowerDecompositionIsApplied) {
    auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, param_shape);
    auto function = createInitGraph(param, 2);
    auto function_ref = createRefGraph(param_shape);

    registerAndRunPass(function);

    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TEST(MVN6PowerDecompositionTest, CheckMVN6PowerDecompositionIsNotApplied) {
    auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, param_shape);
    auto function = createInitGraph(param, 3);
    auto function_ref = createInitGraph(param, 3);

    registerAndRunPass(function);

    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}
