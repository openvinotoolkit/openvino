// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <ngraph_functions/builders.hpp>
#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

using TensorType  = ngraph::element::Type;
using TensorShape = ngraph::Shape;

struct BroadcastInputParams {
    DataShapeWithUpperBound inputShape;
    DataShapeWithUpperBound targetShape;
    InferenceEngine::SizeVector axesMapping;
};

using BroadcastTestParams = std::tuple<
        BroadcastInputParams, TensorType, LayerTestsUtils::TargetDevice>;


class NonZero_Broadcast : public testing::WithParamInterface<BroadcastTestParams>,
                          public DSR_TestsCommon {
protected:
    size_t getDynamicAxis(const DataShape& shapeA, const DataShape& shapeB) const {
        size_t res = 0;
        while (shapeA[res] == shapeB[res]) {
            res++;
        }
        return res;
    }

    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& broadcastParams = std::get<0>(parameters);
        const auto& tensorType = std::get<1>(parameters);
        targetDevice = std::get<2>(GetParam());

        const auto& upperBoundShape = broadcastParams.targetShape.upperBoundShape;
        const auto& realShape = broadcastParams.targetShape.shape;

        const auto dynamicAxis = getDynamicAxis(upperBoundShape, realShape);

        const auto& nonZeroParam = createParameter(tensorType, TensorShape{upperBoundShape[dynamicAxis]});
        const auto& nonZero = std::make_shared<ngraph::opset5::NonZero>(nonZeroParam, ngraph::element::i32);
        m_additionalResults.push_back(std::make_shared<ngraph::opset3::Result>(nonZero->output(0)));
        const auto shapeOfNonZero = std::make_shared<ngraph::opset5::ShapeOf>(nonZero, ngraph::element::i32);
        const auto numNonZeros = std::make_shared<ngraph::opset5::Gather>(
            shapeOfNonZero,
            ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1}),
            ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));

        std::shared_ptr<ngraph::Node> broadcastTargetShape = numNonZeros;

        if (dynamicAxis > 0) {
            broadcastTargetShape = std::make_shared<ngraph::opset5::Concat>(
                ngraph::NodeVector{
                    ngraph::opset5::Constant::create(
                        ngraph::element::i32,
                        ngraph::Shape{dynamicAxis},
                        std::vector<size_t>{upperBoundShape.begin(), upperBoundShape.begin() + dynamicAxis}),
                    broadcastTargetShape},
                0);
        }

        if (dynamicAxis < upperBoundShape.size() - 1) {
            broadcastTargetShape = std::make_shared<ngraph::opset5::Concat>(
                ngraph::NodeVector{
                    broadcastTargetShape,
                    ngraph::opset5::Constant::create(
                        ngraph::element::i32,
                        ngraph::Shape{upperBoundShape.size() - dynamicAxis - 1},
                        std::vector<size_t>{upperBoundShape.begin() + dynamicAxis + 1, upperBoundShape.end()})},
                0);
        }

        const auto& broadcastInput = broadcastParams.inputShape.upperBoundShape.size() ?
            createInputSubgraphWithDSR(tensorType, broadcastParams.inputShape) :
            ngraph::builder::makeConstant(tensorType, ngraph::Shape{broadcastParams.inputShape.shape}, std::vector<int64_t>{}, true);

        if (broadcastParams.axesMapping.size() != 0) {
            const auto& axesMapping = std::get<0>(GetParam()).axesMapping;
            const auto axesMappingConst = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{axesMapping.size()}, axesMapping);

            return std::make_shared<ngraph::opset5::Broadcast>(broadcastInput, broadcastTargetShape, axesMappingConst);
        }

        return std::make_shared<ngraph::opset5::Broadcast>(broadcastInput, broadcastTargetShape, ngraph::op::BroadcastType::BIDIRECTIONAL);
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        if (info.name() == m_parameterVector.front()->get_friendly_name()) {
            // We emulate dynamic target shape through the number of non-zeros in NonZero input tensor
            const auto &broadcastParams = std::get<0>(GetParam());
            const auto numNonZeros = broadcastParams.targetShape.shape[getDynamicAxis(
                    broadcastParams.targetShape.upperBoundShape,
                    broadcastParams.targetShape.shape)];

            auto tensorDesc = info.getTensorDesc();
            auto blob = make_blob_with_precision(tensorDesc);
            blob->allocate();
            CommonTestUtils::fill_data_const(blob, 0);

            InferenceEngine::SizeVector newDims = {numNonZeros};
            blob->getTensorDesc().setDims(newDims);
            CommonTestUtils::fill_data_const(blob, 1);

            blob->getTensorDesc().setDims(tensorDesc.getDims());

            return blob;
        }

        return DSR_TestsCommon::GenerateInput(info);
    }
};

TEST_P(NonZero_Broadcast, CompareWithReference) {
    Run();
}

std::vector<BroadcastInputParams> broadcastTestParams = {
        { DataShapeWithUpperBound{ {1, 1, 4}, {} }, DataShapeWithUpperBound{ {200, 2, 4}, {300, 2, 4} }, {} },
        { DataShapeWithUpperBound{ {15, 14}, {} }, DataShapeWithUpperBound{ {2, 16, 1, 14}, {2, 16, 15, 14} }, {} },
        { DataShapeWithUpperBound{ {15, 1}, {} }, DataShapeWithUpperBound{ {1, 16, 15, 14}, {2, 16, 15, 14} }, {} },
        { DataShapeWithUpperBound{ {2, 16, 15, 14}, {} }, DataShapeWithUpperBound{ {1, 15, 14}, {16, 15, 14} }, {} },
        { DataShapeWithUpperBound{ {2, 16, 15, 14}, {} }, DataShapeWithUpperBound{ {16,  1,  1}, {16,  1,  14}}, {} },
        { DataShapeWithUpperBound{ {16, 15, 1}, {} }, DataShapeWithUpperBound{ {2, 1, 15, 14}, {2, 16, 15, 14} }, {} },
        { DataShapeWithUpperBound{ {142, 1, 1, 64}, {300, 1, 1, 64} }, DataShapeWithUpperBound { {142, 3, 64, 64}, {300, 3, 64, 64} }, {} },
        { DataShapeWithUpperBound{ {1}, {} }, DataShapeWithUpperBound{ {1, 800}, {1, 1000} }, {0} },
        { DataShapeWithUpperBound{ {4}, {} }, DataShapeWithUpperBound{ {100, 4}, {1000, 4} }, {1} },
        { DataShapeWithUpperBound{ {128, 256}, {} }, DataShapeWithUpperBound{ {1, 128, 256}, {3, 128, 256} }, {1, 2} },
};

INSTANTIATE_TEST_SUITE_P(smoke_DynamicBroadcast, NonZero_Broadcast,
        ::testing::Combine(
            ::testing::ValuesIn(broadcastTestParams),
            ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));
}  // namespace
