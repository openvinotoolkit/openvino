// Copyright (C) 2018-2025 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <gtest/gtest.h>

#include <openvino/opsets/opset1.hpp>
#include <transformations/cpu_opset/arm/pass/convert_reduce_no_keep_dims.hpp>
#include "common_test_utils/ov_test_utils.hpp"

using namespace ov::intel_cpu;

template <class T>
class ConvertReduceNoKeepDimsTest : public testing::Test {};

template <class T>
static std::shared_ptr<ov::Model> createInitGraph(std::shared_ptr<ov::opset1::Parameter> param) {
        auto axes = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        auto reduce = std::make_shared<T>(param, axes, false);
        return std::make_shared<ov::Model>(ov::NodeVector{ reduce }, ov::ParameterVector{ param });
}

template <class T>
static std::shared_ptr<ov::Model> createRefGraph(std::shared_ptr<ov::opset1::Parameter> param) {
        auto axes = ov::opset1::Constant::create(ov::element::i64, ov::Shape{2}, {0, 1});
        auto reduce = std::make_shared<T>(param, axes, true);
        auto squeeze = std::make_shared<ov::opset1::Squeeze>(reduce, axes);
        return std::make_shared<ov::Model>(ov::NodeVector{ squeeze }, ov::ParameterVector{ param });
}

template <class T>
static bool registerAndRunReducePass(std::shared_ptr<ov::Model> model) {
    ov::pass::Manager manager;
    if (std::is_base_of<ov::op::util::LogicalReductionKeepDims, T>::value) {
        manager.register_pass<ConvertReduction<ov::op::util::LogicalReductionKeepDims>>();
    } else if (std::is_base_of<ov::op::util::ArithmeticReductionKeepDims, T>::value) {
        manager.register_pass<ConvertReduction<ov::op::util::ArithmeticReductionKeepDims>>();
    } else {
        return false;
    }
    manager.run_passes(model);
    return true;
}

static ov::Shape static_param_shape = ov::Shape{2, 19, 2, 9};
static ov::PartialShape dynamic_param_shape = ov::PartialShape{2, -1, 2, 9};

TYPED_TEST_SUITE_P(ConvertReduceNoKeepDimsTest);

TYPED_TEST_P(ConvertReduceNoKeepDimsTest, CheckConvertReduceTransformationIsAppliedForStaticShapes) {
    ov::element::Type_t dataType = std::is_base_of<ov::op::util::LogicalReductionKeepDims, TypeParam>::value ?
                                   ov::element::boolean : ov::element::f32;
    auto param = std::make_shared<ov::opset1::Parameter>(dataType, static_param_shape);
    auto model = createInitGraph<TypeParam>(param);
    auto model_ref = createRefGraph<TypeParam>(param);

    if (!registerAndRunReducePass<TypeParam>(model)) {
        FAIL() << "Reduce pass is not registered.";
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

TYPED_TEST_P(ConvertReduceNoKeepDimsTest, CheckConvertReduceTransformationIsAppliedForDynaimcShapes) {
    ov::element::Type_t dataType = std::is_base_of<ov::op::util::LogicalReductionKeepDims, TypeParam>::value ?
                                   ov::element::boolean : ov::element::f32;
    auto param = std::make_shared<ov::opset1::Parameter>(dataType, dynamic_param_shape);
    auto model = createInitGraph<TypeParam>(param);
    auto model_ref = createRefGraph<TypeParam>(param);

    if (!registerAndRunReducePass<TypeParam>(model)) {
        FAIL() << "Reduce pass is not registered.";
    }

    auto res = compare_functions(model, model_ref);
    ASSERT_TRUE(res.first) << res.second;
}

REGISTER_TYPED_TEST_SUITE_P(ConvertReduceNoKeepDimsTest,
                            CheckConvertReduceTransformationIsAppliedForStaticShapes,
                            CheckConvertReduceTransformationIsAppliedForDynaimcShapes);

using reduceTypes = ::testing::Types<ov::opset1::ReduceMin,
                                     ov::opset1::ReduceMax,
                                     ov::opset1::ReduceSum,
                                     ov::opset1::ReduceProd,
                                     ov::opset1::ReduceMean,
                                     ov::opset1::ReduceLogicalAnd,
                                     ov::opset1::ReduceLogicalOr>;
INSTANTIATE_TYPED_TEST_SUITE_P(ConvertReduce, ConvertReduceNoKeepDimsTest, reduceTypes);
