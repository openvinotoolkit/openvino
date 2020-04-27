// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <functional_test_utils/layer_test_utils.hpp>
#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

using Parameters = std::tuple<
    DataType,
    DataDims,
    DataDims,
    ngraph::NodeTypeInfo,
    LayerTestsUtils::TargetDevice
>;

class DSR_BinaryElementwise : public testing::WithParamInterface<Parameters>,
        public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& dataDims0 = std::get<1>(parameters);
        const auto& dataDims1 = std::get<2>(parameters);
        const auto& eltwiseType = std::get<3>(parameters);
        targetDevice = std::get<4>(parameters);

        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims0.size()}, dataDims0);
        const auto input1_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size()}, dataDims1);

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_const);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input1, input1_const);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, dsr1});

        function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{eltwise},
            ngraph::ParameterVector{input0, input1},
            eltwiseType.name);
    }
};

class DSR_BinaryElementwiseSingleDSR : public testing::WithParamInterface<Parameters>,
        public LayerTestsUtils::LayerTestsCommon {
protected:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& dataDims0 = std::get<1>(parameters);
        const auto& dataDims1 = std::get<2>(parameters);
        const auto& eltwiseType = std::get<3>(parameters);
        targetDevice = std::get<4>(parameters);

        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims0.size()}, dataDims0);
        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_const);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, input1});

        function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{eltwise},
            ngraph::ParameterVector{input0, input1},
            eltwiseType.name);
    }
};

TEST_P(DSR_BinaryElementwise, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicBinaryElementwise, DSR_BinaryElementwise,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::Values(ngraph::Shape{1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1, 1}),
        ::testing::Values(ngraph::Shape{100}, ngraph::Shape{100, 1}, ngraph::Shape{100, 100}),
        ::testing::Values(ngraph::opset3::Add::type_info,
                          ngraph::opset3::Multiply::type_info,
                          ngraph::opset3::Divide::type_info,
                          ngraph::opset3::Subtract::type_info,
//                        ngraph::opset3::Equal::type_info, operation broadcast default value needs to be fixed
                          ngraph::opset3::Power::type_info),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

TEST_P(DSR_BinaryElementwiseSingleDSR, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_CASE_P(DISABLED_DynamicBinaryElementwiseSingleDSR, DSR_BinaryElementwiseSingleDSR,
    ::testing::Combine(
        ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
        ::testing::Values(ngraph::Shape{1}, ngraph::Shape{1, 1}, ngraph::Shape{1, 1, 1}),
        ::testing::Values(ngraph::Shape{100}, ngraph::Shape{100, 1}, ngraph::Shape{100, 100}),
        ::testing::Values(ngraph::opset3::Add::type_info,
                          ngraph::opset3::Multiply::type_info,
                          ngraph::opset3::Divide::type_info,
                          ngraph::opset3::Subtract::type_info,
//                        ngraph::opset3::Equal::type_info, operation broadcast default value needs to be fixed
                          ngraph::opset3::Power::type_info),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
