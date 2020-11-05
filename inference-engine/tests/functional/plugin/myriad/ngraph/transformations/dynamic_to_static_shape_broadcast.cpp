// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_broadcast.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"
#include "vpu/ngraph/operations/static_shape_broadcast.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"

#include <ngraph_functions/utils/ngraph_helpers.hpp>
#include <ngraph/function.hpp>
#include <ngraph/opsets/opset3.hpp>
#include <ngraph/opsets/opset5.hpp>

#include <common_test_utils/test_common.hpp>
#include <gtest/gtest.h>

#include <string>
#include <memory>
#include <map>
#include <vector>

namespace {

using TensorType  = ngraph::element::Type;
using TensorShape = ngraph::PartialShape;
using AxesMapping = std::vector<size_t>;

struct BroadcastShapes {
    TensorShape srcShape;
    TensorShape targetShape;
    AxesMapping axesMapping;
};
using BroadcastTestParams = std::tuple<TensorType, BroadcastShapes>;

class DynamicToStaticShapeBroadcastExplicitTests
        : public CommonTestUtils::TestsCommon,
          public testing::WithParamInterface<BroadcastTestParams> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& tensorType  = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters).srcShape;
        const auto& targetShape = std::get<1>(parameters).targetShape;
        const auto& axesMapping = std::get<1>(parameters).axesMapping;

        ngraph::helpers::CompareFunctions(
            *transform(tensorType, tensorShape, targetShape, axesMapping),
            *reference(tensorType, tensorShape, targetShape, axesMapping));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const TensorType& tensorType,
            const TensorShape& tensorShape,
            const TensorShape& targetShape,
            const AxesMapping& axesMapping) const {
        const auto tensorParam = std::make_shared<ngraph::opset3::Parameter>(tensorType, tensorShape);
        const auto tensorWithTargetShapeParam = std::make_shared<ngraph::opset3::Parameter>(tensorType, targetShape);

        const auto shapeOfNode = std::make_shared<ngraph::opset3::ShapeOf>(tensorWithTargetShapeParam);
        shapeOfNode->set_is_foldable(false);

        const auto axesMappingConstant = std::make_shared<ngraph::opset3::Constant>(
            ngraph::element::u64,
            ngraph::Shape{axesMapping.size()},
            axesMapping);

        const auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(tensorParam, shapeOfNode, axesMappingConstant);

        auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{broadcast},
            ngraph::ParameterVector{tensorParam, tensorWithTargetShapeParam},
            "Actual");

        // We need to set broadcast output shape to make its rank static.
        // In opset3::Broadcast implementation with Explicit mode output shape gets
        // static rank only in cases when the second input is Concat
        std::vector<ngraph::Dimension> broadcastOutShape(shapeOfNode->get_output_shape(0)[0], ngraph::Dimension::dynamic());
        broadcast->set_output_type(0, tensorParam->get_output_element_type(0), ngraph::PartialShape(broadcastOutShape));
        function->get_result()->set_output_type(0, tensorParam->get_output_element_type(0), targetShape);

        const auto transformations = vpu::Transformations{{ngraph::opset3::Broadcast::type_info, vpu::dynamicToStaticShapeBroadcast}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const TensorType& tensorType,
            const TensorShape& tensorShape,
            const TensorShape& targetShape,
            const AxesMapping& axesMapping) const {
        const auto tensorParam = std::make_shared<ngraph::opset3::Parameter>(tensorType, tensorShape);
        const auto tensorWithTargetShapeParam = std::make_shared<ngraph::opset3::Parameter>(tensorType, targetShape);
        const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(tensorWithTargetShapeParam);

        const auto axesMappingConstant = std::make_shared<ngraph::opset3::Constant>(
            ngraph::element::u64,
            ngraph::Shape{axesMapping.size()},
            axesMapping);

        const auto staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(tensorParam, shapeOf, axesMappingConstant);

        const auto dsrOut = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(staticShapeBroadcast, shapeOf);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsrOut},
            ngraph::ParameterVector{tensorParam, tensorWithTargetShapeParam},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeBroadcastExplicitTests, compareFunctions) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticShapeBroadcastExplicitTests, testing::Combine(
        testing::Values(
            ngraph::element::f16,
            ngraph::element::f32,
            ngraph::element::i32,
            ngraph::element::i64,
            ngraph::element::u8),
        testing::Values(
            BroadcastShapes{TensorShape{16}, TensorShape{1, 16, 50, 50}, AxesMapping{1}},
            BroadcastShapes{TensorShape{50, 50}, TensorShape{1, 50, 50, 16}, AxesMapping{1, 2}})

));

class DynamicToStaticShapeBroadcastBidirectionalTests : public CommonTestUtils::TestsCommon,
                                                        public testing::WithParamInterface<BroadcastTestParams> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& tensorType  = std::get<0>(parameters);
        const auto& tensorShape = std::get<1>(parameters).srcShape;
        const auto& targetShape = std::get<1>(parameters).targetShape;

        ngraph::helpers::CompareFunctions(
            *transform(tensorType, tensorShape, targetShape),
            *reference(tensorType, tensorShape, targetShape));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const TensorType& tensorType,
            const TensorShape& tensorShape,
            const TensorShape& targetShape) const {
        const auto tensorParam = std::make_shared<ngraph::opset5::Parameter>(tensorType, tensorShape);
        const auto tensorWithTargetShapeParam = std::make_shared<ngraph::opset5::Parameter>(tensorType, targetShape);

        const auto shapeOfNode = std::make_shared<ngraph::opset5::ShapeOf>(tensorWithTargetShapeParam);
        shapeOfNode->set_is_foldable(false);

        const auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(tensorParam, shapeOfNode, ngraph::op::BroadcastType::BIDIRECTIONAL);

        auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{broadcast},
            ngraph::ParameterVector{tensorParam, tensorWithTargetShapeParam},
            "Actual");

        const auto transformations = vpu::Transformations{{ngraph::opset5::Broadcast::type_info, vpu::dynamicToStaticShapeBroadcast}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const TensorType& tensorType,
            const TensorShape& tensorShape,
            const TensorShape& targetShape) const {
        const auto tensorParam = std::make_shared<ngraph::opset5::Parameter>(tensorType, tensorShape);
        const auto tensorWithTargetShapeParam = std::make_shared<ngraph::opset5::Parameter>(tensorType, targetShape);
        std::shared_ptr<ngraph::Node> shapeOf = std::make_shared<ngraph::opset5::ShapeOf>(tensorWithTargetShapeParam);

        const auto staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
            tensorParam,
            shapeOf,
            ngraph::op::BroadcastType::BIDIRECTIONAL);

        const auto tensorShapeDimsCount = tensorShape.rank().get_length();
        const auto targetShapeDimsCount = targetShape.rank().get_length();

        std::shared_ptr<ngraph::Node> tensorShapeConst = std::make_shared<ngraph::opset5::Constant>(
            ngraph::element::i64,
            ngraph::Shape{static_cast<size_t>(tensorShapeDimsCount)},
            tensorShape.get_shape());

        const auto maxRankNode = tensorShapeDimsCount > targetShapeDimsCount ? tensorShapeConst : shapeOf;
        const auto minRankNode = maxRankNode == tensorShapeConst ? shapeOf : tensorShapeConst;
        const auto maxRank = maxRankNode == tensorShapeConst ? tensorShapeDimsCount : targetShapeDimsCount;
        const auto minRank = minRankNode == tensorShapeConst ? tensorShapeDimsCount : targetShapeDimsCount;

        ngraph::NodeVector dims;

        for (int i = 0; i < maxRank - minRank; i++) {
            dims.push_back(
                std::make_shared<ngraph::opset5::Gather>(
                    maxRankNode,
                    ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {i}),
                    ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0})));
        }

        for (int i = 0; i < minRank; i++) {
            const auto minRankDim = std::make_shared<ngraph::opset5::Gather>(
                minRankNode,
                ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {i}),
                ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
            const auto maxRankDim = std::make_shared<ngraph::opset5::Gather>(
                maxRankNode,
                ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {maxRank - minRank + i}),
                ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));
            dims.push_back(std::make_shared<ngraph::opset5::Maximum>(minRankDim, maxRankDim));
        }

        const auto outShape = std::make_shared<ngraph::opset5::Concat>(dims, 0);

        const auto dsrOut = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(
            staticShapeBroadcast->output(0), outShape);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsrOut},
            ngraph::ParameterVector{tensorParam, tensorWithTargetShapeParam},
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeBroadcastBidirectionalTests, compareFunctions) {
}

INSTANTIATE_TEST_CASE_P(smoke_NGraph, DynamicToStaticShapeBroadcastBidirectionalTests, testing::Combine(
        testing::Values(
            ngraph::element::f16,
            ngraph::element::f32,
            ngraph::element::i32,
            ngraph::element::i64,
            ngraph::element::u8),
        testing::Values(
            BroadcastShapes{TensorShape{1, 1, 4}, TensorShape{300, 2, 4}, {}},
            BroadcastShapes{TensorShape{15,  1}, TensorShape{2, 16, 15, 14}, {}},
            BroadcastShapes{TensorShape{2, 16, 15, 14}, TensorShape{15, 14}, {}},
            BroadcastShapes{TensorShape{2, 16, 15, 14}, TensorShape{16,  1,  1}, {}},
            BroadcastShapes{TensorShape{2, 16, 15, 14}, TensorShape{16,  1, 14}, {}},
            BroadcastShapes{TensorShape{16, 15,  1}, TensorShape{2, 1, 15, 14}, {}})

));

}  // namespace
