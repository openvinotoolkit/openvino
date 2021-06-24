// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "vpu/ngraph/transformations/dynamic_to_static_shape_broadcast.hpp"
#include "vpu/ngraph/transformations/dynamic_to_static_shape.hpp"
#include "vpu/ngraph/operations/static_shape_broadcast.hpp"
#include "vpu/ngraph/operations/dynamic_shape_resolver.hpp"
#include "vpu/ngraph/utilities.hpp"

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

enum class BroadcastInputType {
    DYNAMIC,
    STATIC
};

struct BroadcastShapes {
    TensorShape srcShape;
    TensorShape targetShape;
    AxesMapping axesMapping;
};

using BroadcastTestParams = std::tuple<
    TensorType,
    TensorType,
    BroadcastShapes,
    BroadcastInputType>;

class DynamicToStaticShapeBroadcastExplicitTests
        : public CommonTestUtils::TestsCommon,
          public testing::WithParamInterface<BroadcastTestParams> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& tensorType         = std::get<0>(parameters);
        const auto& shapeType          = std::get<1>(parameters);
        const auto& tensorShape        = std::get<2>(parameters).srcShape;
        const auto& targetShape        = std::get<2>(parameters).targetShape;
        const auto& axesMapping        = std::get<2>(parameters).axesMapping;
        const auto& broadcastInputType = std::get<3>(parameters);

        ngraph::helpers::CompareFunctions(
            *transform(tensorType, shapeType, tensorShape, targetShape, axesMapping, broadcastInputType),
            *reference(tensorType, shapeType, tensorShape, targetShape, axesMapping, broadcastInputType));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const TensorType& tensorType,
            const TensorType& shapeType,
            const TensorShape& tensorShape,
            const TensorShape& targetShape,
            const AxesMapping& axesMapping,
            BroadcastInputType broadcastInputType) const {
        const auto tensorParam = std::make_shared<ngraph::opset3::Parameter>(tensorType, tensorShape);
        const auto tensorWithTargetShapeParam = std::make_shared<ngraph::opset3::Parameter>(tensorType, targetShape);

        const auto shapeOfNode = std::make_shared<ngraph::opset3::ShapeOf>(tensorWithTargetShapeParam, shapeType);
        shapeOfNode->set_is_foldable(false);

        ngraph::ParameterVector params{tensorParam, tensorWithTargetShapeParam};

        std::shared_ptr<ngraph::Node> broadcastInput = tensorParam;
        if (broadcastInputType == BroadcastInputType::DYNAMIC) {
            const auto shapeParam = std::make_shared<ngraph::opset5::Parameter>(
                shapeType,
                ngraph::Shape{static_cast<size_t>(tensorShape.rank().get_length())});
            params.push_back(shapeParam);
            broadcastInput = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(tensorParam, shapeParam);
        }

        const auto axesMappingConstant = std::make_shared<ngraph::opset3::Constant>(
            ngraph::element::u64,
            ngraph::Shape{axesMapping.size()},
            axesMapping);

        const auto broadcast = std::make_shared<ngraph::opset3::Broadcast>(broadcastInput, shapeOfNode, axesMappingConstant);

        auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{broadcast},
            params,
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
            const TensorType& shapeType,
            const TensorShape& tensorShape,
            const TensorShape& targetShape,
            const AxesMapping& axesMapping,
            BroadcastInputType broadcastInputType) const {
        const auto tensorParam = std::make_shared<ngraph::opset3::Parameter>(tensorType, tensorShape);
        const auto tensorWithTargetShapeParam = std::make_shared<ngraph::opset3::Parameter>(tensorType, targetShape);
        const auto shapeOf = std::make_shared<ngraph::opset3::ShapeOf>(tensorWithTargetShapeParam, shapeType);

        ngraph::ParameterVector params{tensorParam, tensorWithTargetShapeParam};

        std::shared_ptr<ngraph::Node> broadcastInput = tensorParam;
        if (broadcastInputType == BroadcastInputType::DYNAMIC) {
            const auto shapeParam = std::make_shared<ngraph::opset5::Parameter>(
                shapeType,
                ngraph::Shape{static_cast<size_t>(tensorShape.rank().get_length())});
            params.push_back(shapeParam);
            broadcastInput = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(tensorParam, shapeParam);
        }

        const auto axesMappingConstant = std::make_shared<ngraph::opset3::Constant>(
            ngraph::element::i64,
            ngraph::Shape{axesMapping.size()},
            axesMapping);

        const auto staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(broadcastInput, shapeOf, axesMappingConstant);

        const auto dsrOut = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(staticShapeBroadcast, shapeOf);
        return std::make_shared<ngraph::Function>(
            ngraph::NodeVector{dsrOut},
            params,
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeBroadcastExplicitTests, compareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeBroadcastExplicitTests, testing::Combine(
        testing::Values(
            ngraph::element::f16,
            ngraph::element::f32,
            ngraph::element::i32,
            ngraph::element::i64,
            ngraph::element::u8),
        testing::Values(
            ngraph::element::i32,
            ngraph::element::i64),
        testing::Values(
            BroadcastShapes{TensorShape{16}, TensorShape{1, 16, 50, 50}, AxesMapping{1}},
            BroadcastShapes{TensorShape{50, 50}, TensorShape{1, 50, 50, 16}, AxesMapping{1, 2}}),
        testing::Values(
            BroadcastInputType::DYNAMIC,
            BroadcastInputType::STATIC)

));

class DynamicToStaticShapeBroadcastBidirectionalTests : public CommonTestUtils::TestsCommon,
                                                        public testing::WithParamInterface<BroadcastTestParams> {
public:
    void SetUp() override {
        const auto& parameters = GetParam();
        const auto& tensorType         = std::get<0>(parameters);
        const auto& shapeType          = std::get<1>(parameters);
        const auto& tensorShape        = std::get<2>(parameters).srcShape;
        const auto& targetShape        = std::get<2>(parameters).targetShape;
        const auto& broadcastInputType = std::get<3>(parameters);

        ngraph::helpers::CompareFunctions(
            *transform(tensorType, shapeType, tensorShape, targetShape, broadcastInputType),
            *reference(tensorType, shapeType, tensorShape, targetShape, broadcastInputType));
    }

protected:
    std::shared_ptr<const ngraph::Function> transform(
            const TensorType& tensorType,
            const TensorType& shapeType,
            const TensorShape& tensorShape,
            const TensorShape& targetShape,
            BroadcastInputType broadcastInputType) const {
        const auto tensorParam = std::make_shared<ngraph::opset5::Parameter>(tensorType, tensorShape);
        const auto tensorWithTargetShapeParam = std::make_shared<ngraph::opset5::Parameter>(shapeType, targetShape);

        const auto shapeOfNode = std::make_shared<ngraph::opset5::ShapeOf>(tensorWithTargetShapeParam, shapeType);
        shapeOfNode->set_is_foldable(false);

        ngraph::ParameterVector params{tensorParam, tensorWithTargetShapeParam};

        std::shared_ptr<ngraph::Node> broadcastInput = tensorParam;
        if (broadcastInputType == BroadcastInputType::DYNAMIC) {
            const auto shapeParam = std::make_shared<ngraph::opset5::Parameter>(
                shapeType,
                ngraph::Shape{static_cast<size_t>(tensorShape.rank().get_length())});
            params.push_back(shapeParam);
            broadcastInput = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(tensorParam, shapeParam);
        }

        const auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(broadcastInput, shapeOfNode, ngraph::op::BroadcastType::BIDIRECTIONAL);
        // tests are invalid -- output shape of broadcast gets fully deduced and transformations stop working for this particular graph
        broadcast->set_output_type(0, broadcast->get_output_element_type(0), ngraph::PartialShape::dynamic(broadcast->get_output_partial_shape(0).rank()));
        auto function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{broadcast},
            params,
            "Actual");

        const auto transformations = vpu::Transformations{{ngraph::opset5::Broadcast::type_info, vpu::dynamicToStaticShapeBroadcast}};
        vpu::DynamicToStaticShape(transformations).run_on_function(function);
        return function;
    }

    std::shared_ptr<const ngraph::Function> reference(
            const TensorType& tensorType,
            const TensorType& shapeType,
            const TensorShape& tensorShape,
            const TensorShape& targetShape,
            BroadcastInputType broadcastInputType) const {
        const auto tensorParam = std::make_shared<ngraph::opset5::Parameter>(tensorType, tensorShape);
        const auto tensorWithTargetShapeParam = std::make_shared<ngraph::opset5::Parameter>(shapeType, targetShape);
        std::shared_ptr<ngraph::Node> shapeOf = std::make_shared<ngraph::opset5::ShapeOf>(tensorWithTargetShapeParam, shapeType);

        ngraph::ParameterVector params{tensorParam, tensorWithTargetShapeParam};

        std::shared_ptr<ngraph::Node> broadcastInput = tensorParam;
        if (broadcastInputType == BroadcastInputType::DYNAMIC) {
            const auto shapeParam = std::make_shared<ngraph::opset5::Parameter>(
                shapeType,
                ngraph::Shape{static_cast<size_t>(tensorShape.rank().get_length())});
            params.push_back(shapeParam);
            broadcastInput = std::make_shared<ngraph::vpu::op::DynamicShapeResolver>(tensorParam, shapeParam);
        }

        const auto staticShapeBroadcast = std::make_shared<ngraph::vpu::op::StaticShapeBroadcast>(
            broadcastInput,
            shapeOf,
            ngraph::op::BroadcastType::BIDIRECTIONAL);

        const auto tensorShapeDimsCount = tensorShape.rank().get_length();
        const auto targetShapeDimsCount = targetShape.rank().get_length();

        const auto tensorShapeNode = broadcastInputType == BroadcastInputType::DYNAMIC ?
            staticShapeBroadcast->input_value(0).get_node_shared_ptr()->input_value(1) :
            vpu::shapeToConstant(shapeType, tensorShape.get_shape());

        const auto maxRankNode = tensorShapeDimsCount > targetShapeDimsCount ? tensorShapeNode : shapeOf;
        const auto minRankNode = maxRankNode == tensorShapeNode ? shapeOf : tensorShapeNode;
        const auto maxRank = maxRankNode == tensorShapeNode ? tensorShapeDimsCount : targetShapeDimsCount;
        const auto minRank = minRankNode == tensorShapeNode ? tensorShapeDimsCount : targetShapeDimsCount;

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
            params,
            "Expected");
    }
};

TEST_P(DynamicToStaticShapeBroadcastBidirectionalTests, compareFunctions) {
}

INSTANTIATE_TEST_SUITE_P(smoke_NGraph, DynamicToStaticShapeBroadcastBidirectionalTests, testing::Combine(
        testing::Values(
            ngraph::element::f16,
            ngraph::element::f32,
            ngraph::element::i32,
            ngraph::element::i64,
            ngraph::element::u8),
        testing::Values(
                ngraph::element::i32,
                ngraph::element::i64),
        testing::Values(
            BroadcastShapes{TensorShape{1, 1, 4}, TensorShape{300, 2, 4}, {}},
            BroadcastShapes{TensorShape{15,  1}, TensorShape{2, 16, 15, 14}, {}},
            BroadcastShapes{TensorShape{2, 16, 15, 14}, TensorShape{15, 14}, {}},
            BroadcastShapes{TensorShape{2, 16, 15, 14}, TensorShape{16,  1,  1}, {}},
            BroadcastShapes{TensorShape{2, 16, 15, 14}, TensorShape{16,  1, 14}, {}},
            BroadcastShapes{TensorShape{16, 15,  1}, TensorShape{2, 1, 15, 14}, {}}),
        testing::Values(
            BroadcastInputType::DYNAMIC,
            BroadcastInputType::STATIC)
));

}  // namespace
