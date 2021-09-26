// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include <common_test_utils/test_common.hpp>

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape.hpp>
#include <vpu/ngraph/transformations/dynamic_to_static_shape_gather_nd.hpp>
#include <vpu/ngraph/utilities.hpp>

#include <ngraph_functions/utils/ngraph_helpers.hpp>

#include <ngraph/shape.hpp>
#include <ngraph/type/element_type.hpp>
#include <ngraph/opsets/opset5.hpp>

#include <numeric>

namespace {

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

using GatherNDInputsSetup = std::tuple<
    ngraph::ParameterVector, // function parameters
    std::shared_ptr<ngraph::Node>, // data node
    std::shared_ptr<ngraph::Node>, // indices node
    std::shared_ptr<ngraph::Node>, // data shape
    std::shared_ptr<ngraph::Node>>; // indices shape

enum class GatherNDTestMode {
    DYNAMIC_DATA,
    DYNAMIC_INDICES,
    ALL_DYNAMIC
};

struct GatherNDTestCase {
    int64_t batchDims;
    ngraph::Shape dataShape, indicesShape;
};


const auto combinations = testing::Combine(
        testing::Values(
                ngraph::element::f16,
                ngraph::element::f32,
                ngraph::element::i32,
                ngraph::element::i64),
        testing::Values(
                ngraph::element::i32,
                ngraph::element::i64),
        testing::Values(
                GatherNDTestCase{0, {1000, 256, 10, 15}, {25, 125, 3}},
                GatherNDTestCase{2, {30, 2, 100, 35}, {30, 2, 3, 1}},
                GatherNDTestCase{0, {3, 3}, {2, 2}},
                GatherNDTestCase{0, {3, 5}, {2, 1}},
                GatherNDTestCase{0, {4, 3, 6}, {2, 1, 2}},
                GatherNDTestCase{1, {2, 2, 2, 2}, {2, 1}},
                GatherNDTestCase{2, {2, 2, 2, 2}, {2, 2, 1}},
                GatherNDTestCase{0, {1, 22743, 4}, {0, 2}}),
        testing::Values(GatherNDTestMode::DYNAMIC_DATA, GatherNDTestMode::DYNAMIC_INDICES, GatherNDTestMode::ALL_DYNAMIC));

class DynamicToStaticShapeGatherND : public CommonTestUtils::TestsCommon,
                                     public testing::WithParamInterface<std::tuple<DataType, DataType, GatherNDTestCase, GatherNDTestMode>> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& dataType = std::get<0>(parameters);
        const auto& indicesType = std::get<1>(parameters);
        const auto& gatherNDSetup = std::get<2>(parameters);
        const auto& gatherNDTestMode = std::get<3>(parameters);

        ngraph::helpers::CompareFunctions(*transform(dataType, indicesType, gatherNDSetup, gatherNDTestMode),
                                          *reference(dataType, indicesType, gatherNDSetup, gatherNDTestMode));
    }

protected:
    GatherNDInputsSetup setupGatherNDInputs(
            const ngraph::element::Type_t& dataType,
            const ngraph::element::Type_t& indicesType,
            const GatherNDTestCase& gatherNDSetup,
            const GatherNDTestMode& testMode) const {
       ngraph::ParameterVector params = {std::make_shared<ngraph::opset5::Parameter>(dataType, gatherNDSetup.dataShape),
                                         std::make_shared<ngraph::opset5::Parameter>(indicesType, gatherNDSetup.indicesShape)};

       std::shared_ptr<ngraph::Node> inputNode, indicesNode, inputShape, indicesShape;

        if (testMode == GatherNDTestMode::DYNAMIC_DATA || testMode == GatherNDTestMode::ALL_DYNAMIC) {
            params.push_back(std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{gatherNDSetup.dataShape.size()}));
            inputNode = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(params[0], params.back());
            inputShape = params.back();
        } else {
            inputNode = params[0];
            inputShape = ngraph::opset5::Constant::create(ngraph::element::i64, {gatherNDSetup.dataShape.size()}, gatherNDSetup.dataShape);
        }

        if (testMode == GatherNDTestMode::DYNAMIC_INDICES || testMode == GatherNDTestMode::ALL_DYNAMIC) {
            params.push_back(std::make_shared<ngraph::opset5::Parameter>(ngraph::element::i64, ngraph::Shape{gatherNDSetup.indicesShape.size()}));
            indicesNode = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(params[1], params.back());
            indicesShape = params.back();
        } else {
            indicesNode = params[1];
            indicesShape = ngraph::opset5::Constant::create(ngraph::element::i64, {gatherNDSetup.indicesShape.size()}, gatherNDSetup.indicesShape);
        }

        return GatherNDInputsSetup{params, inputNode, indicesNode, inputShape, indicesShape};
    }

    std::shared_ptr<const ngraph::Function> transform(
            const ngraph::element::Type_t& dataType,
            const ngraph::element::Type_t& indicesType,
            const GatherNDTestCase& gatherNDSetup,
            const GatherNDTestMode& testMode) const {
        const auto gatherNDInputsSetup = setupGatherNDInputs(dataType, indicesType, gatherNDSetup, testMode);
        const auto& dataNode = std::get<1>(gatherNDInputsSetup);
        const auto& indicesNode = std::get<2>(gatherNDInputsSetup);

        const auto node = std::make_shared<ngraph::opset5::GatherND>(dataNode, indicesNode, gatherNDSetup.batchDims);

        auto outputShape = node->get_output_partial_shape(0);
        const auto function = std::make_shared<ngraph::Function>(
                ngraph::NodeVector{node},
                std::get<0>(gatherNDInputsSetup),
                "Actual");
        node->set_output_type(0, dataType, ngraph::PartialShape::dynamic(1));

        const auto transformations = vpu::Transformations{{node->type_info, vpu::dynamicToStaticShapeGatherND}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const ngraph::element::Type_t& dataType,
            const ngraph::element::Type_t& indicesType,
            const GatherNDTestCase& gatherNDSetup,
            const GatherNDTestMode& testMode) const {
        const auto gatherNDInputsSetup = setupGatherNDInputs(dataType, indicesType, gatherNDSetup, testMode);
        const auto& dataNode = std::get<1>(gatherNDInputsSetup);
        const auto& indicesNode = std::get<2>(gatherNDInputsSetup);
        const auto& dataShape = std::get<3>(gatherNDInputsSetup);
        const auto& indicesShape = std::get<4>(gatherNDInputsSetup);

        const auto node = std::make_shared<ngraph::opset5::GatherND>(dataNode, indicesNode, gatherNDSetup.batchDims);

        const auto dataDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(dataNode);
        const auto indicesDSR = ngraph::as_type_ptr<ngraph::vpu::op::DynamicShapeResolver>(indicesNode);

        const auto dataShapeRank = gatherNDSetup.dataShape.size();
        const auto indicesShapeRank = gatherNDSetup.indicesShape.size();

        std::shared_ptr<ngraph::Node> outputShape;

        if (gatherNDSetup.batchDims > 0) {
            outputShape = std::make_shared<ngraph::opset5::ReduceProd>(
                vpu::gatherShapeElements(indicesShape, 0, gatherNDSetup.batchDims),
                ngraph::opset5::Constant::create(ngraph::element::i64, {}, {0}),
                true);
        }

        if (indicesShapeRank - gatherNDSetup.batchDims - 1 > 0) {
            const auto indicesShapePart = vpu::gatherShapeElements(
                indicesShape,
                gatherNDSetup.batchDims,
                indicesShapeRank - gatherNDSetup.batchDims - 1);
            outputShape = outputShape ? std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector{outputShape, indicesShapePart}, 0) : indicesShapePart;
        }

        const auto lastIndicesDim = node->get_input_partial_shape(1)[indicesShapeRank - 1].get_length();
        if (gatherNDSetup.batchDims + lastIndicesDim < dataShapeRank) {
            const auto dataShapePart = vpu::gatherShapeElements(
                dataShape,
                lastIndicesDim + gatherNDSetup.batchDims,
                dataShapeRank - gatherNDSetup.batchDims - lastIndicesDim);
            outputShape = outputShape ? std::make_shared<ngraph::opset5::Concat>(ngraph::NodeVector{outputShape, dataShapePart}, 0) : dataShapePart;
        }

        const auto outputDsr = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(node, outputShape);
        return std::make_shared<ngraph::Function>(
                ngraph::NodeVector{outputDsr},
                std::get<0>(gatherNDInputsSetup),
                "Expected");
    }
};

TEST_P(DynamicToStaticShapeGatherND, CompareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeGatherND, combinations);

} // namespace
