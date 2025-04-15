// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <string>
#include <memory>

#include <openvino/core/model.hpp>
#include <openvino/opsets/opset1.hpp>
#include <openvino/opsets/opset3.hpp>
#include <openvino/opsets/opset7.hpp>
#include <transformations/cpu_opset/arm/pass/convert_reduce_multi_axis.hpp>
#include <transformations/init_node_info.hpp>
#include <transformations/utils/utils.hpp>
#include <openvino/pass/manager.hpp>
#include <ov_ops/type_relaxed.hpp>

#include "common_test_utils/ov_test_utils.hpp"

using namespace testing;
using namespace ov::intel_cpu;

template <class T>
class ConvertReduceMultiAxisTest : public testing::Test {};

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(std::shared_ptr<ov::opset1::Parameter> param) {
        auto axes = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        auto reduce = std::make_shared<T>(param, axes, true);
        return std::make_shared<ov::Model>(ov::NodeVector{ reduce }, ov::ParameterVector{ param });
}

template <class T>
static std::shared_ptr<ov::Model> createRefGraph(ov::Shape param_shape) {
        auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, param_shape);
        std::vector<int64_t> axes = {0, 1};
        std::shared_ptr<ov::Node> node = param;
        for (auto axis : axes) {
            auto reduction_axis = ov::opset1::Constant::create(ov::element::i64, ov::Shape{}, {axis});
            node = std::make_shared<T>(node, reduction_axis, true);
        }

        return std::make_shared<ov::Model>(ov::NodeVector{ node }, ov::ParameterVector{ param });
}

template <class T>
static bool registerAndRunReducePass(std::shared_ptr<ov::Model> model) {
    ov::pass::Manager manager;
    if (std::is_same<T, ov::opset1::ReduceMin>::value) {
        manager.register_pass<ConvertReduceMin>();
    } else if (std::is_same<T, ov::opset1::ReduceMax>::value) {
        manager.register_pass<ConvertReduceMax>();
    } else if (std::is_same<T, ov::opset1::ReduceSum>::value) {
        manager.register_pass<ConvertReduceSum>();
    } else if (std::is_same<T, ov::opset1::ReduceProd>::value) {
        manager.register_pass<ConvertReduceProd>();
    } else {
        return false;
    }
    manager.run_passes(model);
    return true;
}

static ov::Shape static_param_shape = ov::Shape{2, 19, 2, 9};
static ov::PartialShape dynamic_param_shape = ov::PartialShape{2, -1, 2, 9};

TYPED_TEST_SUITE_P(ConvertReduceMultiAxisTest);

TYPED_TEST_P(ConvertReduceMultiAxisTest, CheckConvertReduceTransformationIsApplied) {
    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, static_param_shape);
    auto model = createInitGraph<TypeParam>(param);
    auto model_ref = createRefGraph<TypeParam>(static_param_shape);

    if (!registerAndRunReducePass<TypeParam>(model)) {
        FAIL() << "Reduce pass is not registered.";
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TYPED_TEST_P(ConvertReduceMultiAxisTest, CheckConvertReduceTransformationIsNotAppliedForDynaimcShapes) {
    auto param = std::make_shared<ov::opset1::Parameter>(ov::element::f32, dynamic_param_shape);
    auto model = createInitGraph<TypeParam>(param);
    auto model_ref = createInitGraph<TypeParam>(param);

    if (!registerAndRunReducePass<TypeParam>(model)) {
        FAIL() << "Reduce pass is not registered.";
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

REGISTER_TYPED_TEST_SUITE_P(ConvertReduceMultiAxisTest,
                            CheckConvertReduceTransformationIsApplied,
                            CheckConvertReduceTransformationIsNotAppliedForDynaimcShapes);

using reduceTypes = ::testing::Types<ov::opset1::ReduceMin,
                                     ov::opset1::ReduceMax,
                                     ov::opset1::ReduceSum,
                                     ov::opset1::ReduceProd>;
INSTANTIATE_TYPED_TEST_SUITE_P(ConvertReduce, ConvertReduceMultiAxisTest, reduceTypes);
