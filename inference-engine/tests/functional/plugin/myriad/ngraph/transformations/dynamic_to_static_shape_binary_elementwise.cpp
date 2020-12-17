// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>
#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>

#include <common_test_utils/test_common.hpp>

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_binary_elementwise.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/utils/error.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>

namespace {

enum class TestShapeTypes {
    ALL_DYNAMIC,
    SINGLE_DSR
};

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;
using refFunction = std::function<std::shared_ptr<ngraph::Function> (
    const DataType&, const ngraph::NodeTypeInfo&, const DataDims&, const DataDims&, TestShapeTypes)>;
using EltwiseParams = std::tuple<DataDims, DataDims, refFunction>;

class DynamicToStaticShapeEltwise: public CommonTestUtils::TestsCommon, public testing::WithParamInterface<std::tuple<ngraph::element::Type_t,
        ngraph::NodeTypeInfo, EltwiseParams, TestShapeTypes>> {
public:
    void SetUp() override {
        const auto& dataType = std::get<0>(GetParam());
        const auto& eltwiseType = std::get<1>(GetParam());
        const auto& eltwiseParams = std::get<2>(GetParam());
        const auto& testShapeTypes = std::get<3>(GetParam());

        const auto& input0Shape = std::get<0>(eltwiseParams);
        const auto& input1Shape = std::get<1>(eltwiseParams);

        ngraph::helpers::CompareFunctions(*transform(dataType, eltwiseType, input0Shape, input1Shape, testShapeTypes),
                                          *std::get<2>(eltwiseParams)(dataType, eltwiseType, input0Shape, input1Shape, testShapeTypes));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1,
        TestShapeTypes testShapeTypes) const {
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0Dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0Dims);

        ngraph::ParameterVector params{input0, input1, input0Dims};

        std::shared_ptr<ngraph::Node> eltwiseInput1 = input1;
        if (testShapeTypes == TestShapeTypes::ALL_DYNAMIC) {
            const auto input1Dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64,
                                                                                ngraph::Shape{dataDims1.size()});
            eltwiseInput1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input1, input1Dims);
            params.push_back(input1Dims);
        }

        const auto eltwise = buildEltwise(eltwiseType, {dsr0, eltwiseInput1}, params, testShapeTypes);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{eltwise},
            params,
            "Actual");

        eltwise->set_output_type(0, eltwise->get_input_element_type(0), ngraph::PartialShape::dynamic(eltwise->get_output_partial_shape(0).rank()));

        const auto transformations = vpu::Transformations{{eltwiseType, vpu::dynamicToStaticShapeBinaryEltwise}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);

        return function;
    }

public:
    static
    std::shared_ptr<ngraph::Function> reference_simple(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1,
        TestShapeTypes testShapeTypes) {
        // Data flow subgraph
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0Dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0Dims);

        ngraph::ParameterVector params{input0, input1, input0Dims};

        std::shared_ptr<ngraph::Node> dims;
        if (testShapeTypes == TestShapeTypes:: ALL_DYNAMIC) {
            params.push_back(std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims1.size()}));
            dims = params.back();
        } else {
            dims = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size()}, dataDims1);
        }

        std::shared_ptr<ngraph::Node> eltwiseInput1 = input1;
        if (testShapeTypes == TestShapeTypes::ALL_DYNAMIC) {
            eltwiseInput1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input1, dims);
        }

        const auto eltwise = buildEltwise(eltwiseType, {dsr0, eltwiseInput1}, params, testShapeTypes);

        // Shape infer subgraph
        const auto maximum = std::make_shared<ngraph::opset3::Maximum>(input0Dims, dims);
        const auto dsr_final = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(eltwise, maximum);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsr_final},
            params,
            "Actual");

        return function;
    }

    static
    std::shared_ptr<ngraph::Function> reference_broadcast_left(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1,
        TestShapeTypes testShapeTypes) {
        // Data flow subgraph
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0Dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0Dims);

        ngraph::ParameterVector params{input0, input1, input0Dims};

        std::shared_ptr<ngraph::Node> dims;
        if (testShapeTypes == TestShapeTypes::ALL_DYNAMIC) {
            params.push_back(std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims1.size()}));
            dims = params.back();
        } else {
            dims = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size()}, dataDims1);
        }

        std::shared_ptr<ngraph::Node> eltwiseInput1 = input1;
        if (testShapeTypes == TestShapeTypes::ALL_DYNAMIC) {
            eltwiseInput1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input1, dims);
        }

        const auto eltwise = buildEltwise(eltwiseType, {dsr0, eltwiseInput1}, params, testShapeTypes);

        // Shape infer subgraph
        const auto broadcastConst = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size() - dataDims0.size()}, {1});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{broadcastConst, input0Dims}, 0);
        const auto maximum = std::make_shared<ngraph::opset3::Maximum>(concat, dims);
        const auto dsrFinal = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(eltwise, maximum);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsrFinal},
            params,
            "Actual");

        return function;
    }

    static
    std::shared_ptr<ngraph::Function> reference_broadcast_right(
        const ngraph::element::Type_t& dataType,
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::Shape& dataDims0,
        const ngraph::Shape& dataDims1,
        TestShapeTypes testShapeTypes) {
        // Data flow subgraph
        const auto input0 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims0);
        const auto input1 = std::make_shared<ngraph::opset3::Parameter>(dataType, dataDims1);

        const auto input0Dims = std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims0.size()});
        const auto dsr0 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input0, input0Dims);

        ngraph::ParameterVector params{input0, input1, input0Dims};

        std::shared_ptr<ngraph::Node> dims;
        if (testShapeTypes == TestShapeTypes::ALL_DYNAMIC) {
            params.push_back(std::make_shared<ngraph::opset3::Parameter>(ngraph::element::i64, ngraph::Shape{dataDims1.size()}));
            dims = params.back();
        } else {
            dims = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims1.size()}, dataDims1);
        }

        std::shared_ptr<ngraph::Node> eltwiseInput1 = input1;
        if (testShapeTypes == TestShapeTypes::ALL_DYNAMIC) {
            eltwiseInput1 = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(input1, dims);
        }

        const auto eltwise = buildEltwise(eltwiseType, {dsr0, eltwiseInput1}, params, testShapeTypes);

        // Shape infer subgraph
        const auto broadcastConst = ngraph::opset3::Constant::create(ngraph::element::i64, {dataDims0.size() - dataDims1.size()}, {1});
        const auto concat = std::make_shared<ngraph::opset3::Concat>(ngraph::OutputVector{broadcastConst, dims}, 0);
        const auto maximum = std::make_shared<ngraph::opset3::Maximum>(input0Dims, concat);
        const auto dsrFinal = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(eltwise, maximum);

        const auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsrFinal},
            params,
            "Actual");

        return function;
    }

private:
    static
    std::shared_ptr<ngraph::Node> buildEltwise(
        const ngraph::NodeTypeInfo& eltwiseType,
        const ngraph::OutputVector& inputs,
        ngraph::ParameterVector& params,
        TestShapeTypes testShapeTypes) {
        if (eltwiseType == ngraph::opset5::Select::type_info) {
            params.push_back(std::make_shared<ngraph::opset3::Parameter>(
                    ngraph::element::boolean,
                    ngraph::Shape{inputs.front().get_shape()}));
            std::shared_ptr<ngraph::Node> condInput = params.back();
            if (testShapeTypes == TestShapeTypes::ALL_DYNAMIC) {
                params.push_back(std::make_shared<ngraph::opset3::Parameter>(
                    ngraph::element::i64,
                    ngraph::Shape{static_cast<size_t>(inputs.front().get_partial_shape().rank().get_length())}));
                condInput = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(condInput, params.back());
            }
            return ngraph::helpers::getNodeSharedPtr(eltwiseType, {condInput, inputs[0], inputs[1]});
        } else {
            return ngraph::helpers::getNodeSharedPtr(eltwiseType, inputs);
        }
    }
};

TEST_P(DynamicToStaticShapeEltwise, CompareFunctions) {
}

INSTANTIATE_TEST_CASE_P(smoke_EltwiseBroadcast, DynamicToStaticShapeEltwise, testing::Combine(
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
        ngraph::opset3::Subtract::type_info,
        ngraph::opset3::Maximum::type_info,
        ngraph::opset3::Minimum::type_info,
        ngraph::opset3::Less::type_info,
        ngraph::opset5::Select::type_info),
    testing::Values(
        EltwiseParams{DataDims{1000}, DataDims{1}, DynamicToStaticShapeEltwise::reference_simple},
        EltwiseParams{DataDims{1000, 1, 1}, DataDims{1000, 1, 1}, DynamicToStaticShapeEltwise::reference_simple},
        EltwiseParams{DataDims{2, 1000}, DataDims{3, 1, 1}, DynamicToStaticShapeEltwise::reference_broadcast_left},
        EltwiseParams{DataDims{1000, 64}, DataDims{1}, DynamicToStaticShapeEltwise::reference_broadcast_right}),
    testing::Values(TestShapeTypes::ALL_DYNAMIC, TestShapeTypes::SINGLE_DSR)
));

}  // namespace