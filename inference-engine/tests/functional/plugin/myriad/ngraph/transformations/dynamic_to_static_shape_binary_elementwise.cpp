// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

#include <common_test_utils/test_common.hpp>

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_binary_elementwise.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;
using refFunction = std::function<std::shared_ptr<ngraph::Function> (const DataType&, const ngraph::NodeTypeInfo&, const DataDims&, const DataDims&)>;
using EltwiseParams = std::tuple<DataDims, DataDims, refFunction>;

class DynamicToStaticShapeEltwise: public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<ngraph::element::Type_t,
        ngraph::NodeTypeInfo, EltwiseParams>> {
public:
    void SetUp() override {
        const auto& dataType = std::get<0>(GetParam());
        const auto& eltwiseType = std::get<1>(GetParam());
        const auto& eltwiseParams = std::get<2>(GetParam());

        const auto& input0_shape = std::get<0>(eltwiseParams);
        const auto& input1_shape = std::get<1>(eltwiseParams);

        ngraph::helpers::CompareFunctions(*transform(dataType, eltwiseType, input0_shape, input1_shape),
                                          *std::get<2>(eltwiseParams)(dataType, eltwiseType, input0_shape, input1_shape));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1) const {
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto input1_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims1.size()});

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_dsr);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input1, input1_dsr);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, dsr1});

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{eltwise},
            ngraph::ParameterVector{input0, input1, input0_dsr, input1_dsr},
            "Actual");

        eltwise->set_output_type(0, eltwise->get_input_element_type(0), ngraph::PartialShape::dynamic(eltwise->get_output_partial_shape(0).rank()));

        const auto transformations = vpu::Transformations{{eltwiseType, vpu::dynamicToStaticShapeBinaryEltwise}};
        vpu::DynamicToStaticShape(transformations).transform(function);
        return function;
    }

public:
    static
    std::shared_ptr<ngraph::Function> reference_simple(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1) {
        // Data flow subgraph
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto input1_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims1.size()});

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_dsr);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input1, input1_dsr);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, dsr1});

        // Shape infer subgraph
        const auto maximum = std::make_shared<ngraph::opset3::Maximum>(input0_dsr, input1_dsr);
        const auto dsr_final = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(eltwise, maximum);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr_final},
            ngraph::ParameterVector{input0, input1, input0_dsr, input1_dsr},
            "Actual");

        return function;
    }

    static
    std::shared_ptr<ngraph::Function> reference_broadcast_left(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1) {
        // Data flow subgraph
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto input1_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims1.size()});

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_dsr);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input1, input1_dsr);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, dsr1});

        // Shape infer subgraph
        const auto broadcast_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size() - dataDims0.size()}, {1});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{broadcast_const, input0_dsr}, 0);
        const auto maximum = std::make_shared<ngraph::opset3::Maximum>(concat, input1_dsr);
        const auto dsr_final = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(eltwise, maximum);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr_final},
            ngraph::ParameterVector{input0, input1, input0_dsr, input1_dsr},
            "Actual");

        return function;
    }

    static
    std::shared_ptr<ngraph::Function> reference_broadcast_right(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1) {
        // Data flow subgraph
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto input1_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims1.size()});

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_dsr);
        const auto dsr1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input1, input1_dsr);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, dsr1});

        // Shape infer subgraph
        const auto broadcast_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims0.size() - dataDims1.size()}, {1});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{broadcast_const, input1_dsr}, 0);
        const auto maximum = std::make_shared<ngraph::opset3::Maximum>(input0_dsr, concat);
        const auto dsr_final = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(eltwise, maximum);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr_final},
            ngraph::ParameterVector{input0, input1, input0_dsr, input1_dsr},
            "Actual");

        return function;
    }
};

class DynamicToStaticShapeEltwiseSingleDSR: public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<ngraph::element::Type_t,
        ngraph::NodeTypeInfo, EltwiseParams>> {
public:
    void SetUp() override {
        const auto& dataType = std::get<0>(GetParam());
        const auto& eltwiseType = std::get<1>(GetParam());
        const auto& eltwiseParams = std::get<2>(GetParam());

        const auto& input0_shape = std::get<0>(eltwiseParams);
        const auto& input1_shape = std::get<1>(eltwiseParams);

        ngraph::helpers::CompareFunctions(*transform(dataType, eltwiseType, input0_shape, input1_shape),
                                          *std::get<2>(eltwiseParams)(dataType, eltwiseType, input0_shape, input1_shape));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1) const {
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_dsr);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, input1});

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{eltwise},
            ngraph::ParameterVector{input0, input1, input0_dsr},
            "Actual");

        eltwise->set_output_type(0, eltwise->get_input_element_type(0), ngraph::PartialShape::dynamic(eltwise->get_output_partial_shape(0).rank()));

        const auto transformations = vpu::Transformations{{eltwiseType, vpu::dynamicToStaticShapeBinaryEltwise}};
        vpu::DynamicToStaticShape(transformations).transform(function);
        return function;
    }

public:
    static
    std::shared_ptr<ngraph::Function> reference_simple(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1) {
        // Data flow subgraph
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto input1_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size()}, dataDims1);

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_dsr);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, input1});

        // Shape infer subgraph
        const auto maximum = std::make_shared<ngraph::opset3::Maximum>(input0_dsr, input1_const);
        const auto dsr_final = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(eltwise, maximum);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr_final},
            ngraph::ParameterVector{input0, input1, input0_dsr},
            "Actual");

        return function;
    }

    static
    std::shared_ptr<ngraph::Function> reference_broadcast_left(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1) {
        // Data flow subgraph
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto input1_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size()}, dataDims1);

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_dsr);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, input1});

        // Shape infer subgraph
        const auto broadcast_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size() - dataDims0.size()}, {1});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{broadcast_const, input0_dsr}, 0);
        const auto maximum = std::make_shared<ngraph::opset3::Maximum>(concat, input1_const);
        const auto dsr_final = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(eltwise, maximum);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr_final},
            ngraph::ParameterVector{input0, input1, input0_dsr},
            "Actual");

        return function;
    }

    static
    std::shared_ptr<ngraph::Function> reference_broadcast_right(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1) {
        // Data flow subgraph
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0_dsr = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto input1_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size()}, dataDims1);

        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0_dsr);

        const auto eltwise = ngraph::helpers::getNodeSharedPtr(eltwiseType, {dsr0, input1});

        // Shape infer subgraph
        const auto broadcast_const = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims0.size() - dataDims1.size()}, {1});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{broadcast_const, input1_const}, 0);
        const auto maximum = std::make_shared<ngraph::opset3::Maximum>(input0_dsr, concat);
        const auto dsr_final = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(eltwise, maximum);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr_final},
            ngraph::ParameterVector{input0, input1, input0_dsr},
            "Actual");

        return function;
    }
};

TEST_P(DynamicToStaticShapeEltwise, CompareFunctions) {
}

INSTANTIATE_TEST_CASE_P(EltwiseBroadcast, DynamicToStaticShapeEltwise, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        ngraph::opset3::Add::type_info,
        ngraph::opset3::Divide::type_info,
        ngraph::opset3::Equal::type_info,
        ngraph::opset3::Greater::type_info,
        ngraph::opset3::Power::type_info,
        ngraph::opset3::Multiply::type_info,
        ngraph::opset3::Subtract::type_info),
    testing::Values(
        EltwiseParams{DataDims{1000}, DataDims{1}, DynamicToStaticShapeEltwise::reference_simple},
        EltwiseParams{DataDims{1000, 1, 1}, DataDims{1000, 1, 1}, DynamicToStaticShapeEltwise::reference_simple},
        EltwiseParams{DataDims{2, 1000}, DataDims{3, 1, 1}, DynamicToStaticShapeEltwise::reference_broadcast_left},
        EltwiseParams{DataDims{1000, 64}, DataDims{1}, DynamicToStaticShapeEltwise::reference_broadcast_right})));

TEST_P(DynamicToStaticShapeEltwiseSingleDSR, CompareFunctions) {
}

INSTANTIATE_TEST_CASE_P(EltwiseBroadcastSingleDSR, DynamicToStaticShapeEltwiseSingleDSR, testing::Combine(
    testing::Values(
        ngraph::element::f16,
        ngraph::element::f32,
        ngraph::element::i32,
        ngraph::element::i64,
        ngraph::element::u8),
    testing::Values(
        ngraph::opset3::Add::type_info,
        ngraph::opset3::Divide::type_info,
        ngraph::opset3::Equal::type_info,
        ngraph::opset3::Greater::type_info,
        ngraph::opset3::Power::type_info,
        ngraph::opset3::Multiply::type_info,
        ngraph::opset3::Subtract::type_info),
    testing::Values(
        EltwiseParams{DataDims{1000}, DataDims{1}, DynamicToStaticShapeEltwiseSingleDSR::reference_simple},
        EltwiseParams{DataDims{1000, 1, 1}, DataDims{1000, 1, 1}, DynamicToStaticShapeEltwiseSingleDSR::reference_simple},
        EltwiseParams{DataDims{2, 1000}, DataDims{3, 1, 1}, DynamicToStaticShapeEltwiseSingleDSR::reference_broadcast_left},
        EltwiseParams{DataDims{1000, 64}, DataDims{1}, DynamicToStaticShapeEltwiseSingleDSR::reference_broadcast_right})));
}  // namespace