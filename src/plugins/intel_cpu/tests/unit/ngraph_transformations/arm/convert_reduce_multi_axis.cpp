// Copyright (C) 2018-2023 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <ngraph/function.hpp>
#include <ngraph/opsets/opset1.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset7.hpp>
#include <transformations/cpu_opset/arm/pass/convert_reduce_multi_axis.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <ngraph/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ngraph_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

template <class T>
class ConvertReduceMultiAxisTest : public testing::Test {};

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(std::shared_ptr<ngraph::opset1::Parameter> param) {
        auto axes = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{2}, {0, 1});
        auto reduce = std::make_shared<T>(param, axes, true);
        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ reduce }, ngraph::ParameterVector{ param });
}

template <class T>
static std::shared_ptr<ov::Model> createRefGraph(ov::Shape param_shape) {
        auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, param_shape);
        std::vector<int64_t> axes = {0, 1};
        std::shared_ptr<ngraph::Node> node = param;
        for (auto axis : axes) {
            auto reduction_axis = ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{}, {axis});
            node = std::make_shared<T>(node, reduction_axis, true);
        }

        return std::make_shared<ngraph::Function>(ngraph::NodeVector{ node }, ngraph::ParameterVector{ param });
}

template <class T>
static bool registerAndRunReducePass(std::shared_ptr<ov::Model> function) {
    ov::pass::Manager manager;
    if (std::is_same<T, ngraph::opset1::ReduceMin>::value) {
        manager.register_pass<ConvertReduceMin>();
    } else if (std::is_same<T, ngraph::opset1::ReduceMax>::value) {
        manager.register_pass<ConvertReduceMax>();
    } else if (std::is_same<T, ngraph::opset1::ReduceSum>::value) {
        manager.register_pass<ConvertReduceSum>();
    } else if (std::is_same<T, ngraph::opset1::ReduceProd>::value) {
        manager.register_pass<ConvertReduceProd>();
    } else {
        return false;
    }
    manager.run_passes(function);
    return true;
}

static ngraph::Shape static_param_shape = ngraph::Shape{2, 19, 2, 9};
static ngraph::PartialShape dynamic_param_shape = ngraph::PartialShape{2, -1, 2, 9};

TYPED_TEST_SUITE_P(ConvertReduceMultiAxisTest);

TYPED_TEST_P(ConvertReduceMultiAxisTest, CheckConvertReduceTransformationIsApplied) {
    auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, static_param_shape);
    auto function = createInitGraph<TypeParam>(param);
    auto function_ref = createRefGraph<TypeParam>(static_param_shape);

    if (!registerAndRunReducePass<TypeParam>(function)) {
        FAIL() << "Reduce pass is not registered.";
    }

    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TYPED_TEST_P(ConvertReduceMultiAxisTest, CheckConvertReduceTransformationIsNotAppliedForDynaimcShapes) {
    auto param = std::make_shared<ngraph::opset1::Parameter>(ngraph::element::f32, dynamic_param_shape);
    auto function = createInitGraph<TypeParam>(param);
    auto function_ref = createInitGraph<TypeParam>(param);

    if (!registerAndRunReducePass<TypeParam>(function)) {
        FAIL() << "Reduce pass is not registered.";
    }

    auto res = compare_functions(function, function_ref);
    ASSERT_TRUE(res.first) << res.second;
}

REGISTER_TYPED_TEST_SUITE_P(ConvertReduceMultiAxisTest,
                            CheckConvertReduceTransformationIsApplied,
                            CheckConvertReduceTransformationIsNotAppliedForDynaimcShapes);

using reduceTypes = ::testing::Types<ngraph::opset1::ReduceMin,
                                     ngraph::opset1::ReduceMax,
                                     ngraph::opset1::ReduceSum,
                                     ngraph::opset1::ReduceProd>;
INSTANTIATE_TYPED_TEST_SUITE_P(ConvertReduce, ConvertReduceMultiAxisTest, reduceTypes);
