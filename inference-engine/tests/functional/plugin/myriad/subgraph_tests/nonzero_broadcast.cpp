// Copyright (C) 2020 Intel Corporation
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
    TensorShape inputShape;
    DataShapeWithUpperBound targetShape;
    InferenceEngine::SizeVector axesMapping;
};

using BroadcastTestParams = std::tuple<
        BroadcastInputParams, TensorType, LayerTestsUtils::TargetDevice>;


class NonZero_BroadcastBidirectional : public testing::WithParamInterface<BroadcastTestParams>,
                                       public DSR_TestsCommon {
protected:
    void prepareBroadcastInputs() {
        const auto& parameters = GetParam();
        const auto& broadcastParams = std::get<0>(parameters);
        const auto& tensorType = std::get<1>(parameters);
        targetDevice = std::get<2>(GetParam());

        const auto& upperBoundShape = broadcastParams.targetShape.upperBoundShape;
        const auto& realShape = broadcastParams.targetShape.shape;

        while (upperBoundShape[m_dynamicAxis] == realShape[m_dynamicAxis]) {
            m_dynamicAxis++;
        }

        const auto tensorParam = createParameter(tensorType, TensorShape{upperBoundShape[m_dynamicAxis]});
        const auto nonZero = std::make_shared<ngraph::opset5::NonZero>(tensorParam);
        const auto shapeOfNonZero = std::make_shared<ngraph::opset5::ShapeOf>(nonZero);
        const auto numNonZeros = std::make_shared<ngraph::opset5::Gather>(shapeOfNonZero,
                                                                          ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1}),
                                                                          ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));

        m_broadcastTargetShape = numNonZeros;

        if (m_dynamicAxis > 0) {
            m_broadcastTargetShape = std::make_shared<ngraph::opset5::Concat>(
                    ngraph::NodeVector{ngraph::opset5::Constant::create(
                            ngraph::element::i64,
                            ngraph::Shape{m_dynamicAxis},
                            std::vector<size_t>{upperBoundShape.begin(), upperBoundShape.begin() + m_dynamicAxis}),
                                       m_broadcastTargetShape},
                    0);
        }

        if (m_dynamicAxis < upperBoundShape.size() - 1) {
            m_broadcastTargetShape = std::make_shared<ngraph::opset5::Concat>(
                    ngraph::NodeVector{m_broadcastTargetShape,
                                       ngraph::opset5::Constant::create(
                                               ngraph::element::i64,
                                               ngraph::Shape{upperBoundShape.size() - m_dynamicAxis - 1},
                                               std::vector<size_t>{upperBoundShape.begin() + m_dynamicAxis + 1, upperBoundShape.end()})},
                    0);
        }

        m_broadcastInput = ngraph::builder::makeConstant(tensorType, ngraph::Shape{broadcastParams.inputShape}, std::vector<int64_t>{}, true);
        m_additionalResults.push_back(std::make_shared<ngraph::opset5::Result>(nonZero->output(0)));
    }

    std::shared_ptr<ngraph::Node> createTestedOp() override {
        prepareBroadcastInputs();

        const auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(
                m_broadcastInput, m_broadcastTargetShape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        return broadcast;
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        // We emulate dynamic shape through the number of non-zeros in NonZero input tensor
        const auto& broadcastParams = std::get<0>(GetParam());
        const auto numNonZeros = broadcastParams.targetShape.shape[m_dynamicAxis];

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

protected:
    size_t m_dynamicAxis = 0;
    std::shared_ptr<ngraph::Node> m_broadcastInput;
    std::shared_ptr<ngraph::Node> m_broadcastTargetShape;
};

TEST_P(NonZero_BroadcastBidirectional, CompareWithReference) {
    Run();
}

std::vector<BroadcastInputParams> broadcastBidirectionalTestParams = {
        { {1, 1, 4}, DataShapeWithUpperBound{ {200, 2, 4}, {300, 2, 4} }, {} },
        { {15, 14}, DataShapeWithUpperBound{ {2, 16, 1, 14}, {2, 16, 15, 14} }, {} },
        { {15, 1}, DataShapeWithUpperBound{ {1, 16, 15, 14}, {2, 16, 15, 14} }, {} },
        { {2, 16, 15, 14}, DataShapeWithUpperBound{ {1, 15, 14}, {16, 15, 14} }, {} },
        { {2, 16, 15, 14}, DataShapeWithUpperBound{ {16,  1,  1}, {16,  1,  14}}, {} },
        { {16, 15,  1}, DataShapeWithUpperBound{ {2, 1, 15, 14}, {2, 16, 15, 14} }, {} },
};

INSTANTIATE_TEST_CASE_P(smoke_DynamicBroadcast, NonZero_BroadcastBidirectional,
        ::testing::Combine(
            ::testing::ValuesIn(broadcastBidirectionalTestParams),
            ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

using BroadcastExplicitTestParams = std::tuple<
        BroadcastTestParams, TensorShape, TensorType, LayerTestsUtils::TargetDevice>;

class NonZero_BroadcastExplicit : public NonZero_BroadcastBidirectional {
protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        prepareBroadcastInputs();
        const auto& axesMapping = std::get<0>(GetParam()).axesMapping;

        const auto axesMappingConst = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{axesMapping.size()}, axesMapping);

        const auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(
                m_broadcastInput, m_broadcastTargetShape, axesMappingConst);

        return broadcast;
    }
};

TEST_P(NonZero_BroadcastExplicit, CompareWithReference) {
    Run();
}

std::vector<BroadcastInputParams> broadcastExplicitTestParams = {
        { {1}, DataShapeWithUpperBound{ {1, 800}, {1, 1000} }, {0} },
        { {4}, DataShapeWithUpperBound{ {100, 4}, {1000, 4} }, {1} },
        { {128, 256}, DataShapeWithUpperBound{ {1, 128, 256}, {3, 128, 256} }, {1, 2} },
};

INSTANTIATE_TEST_CASE_P(smoke_DynamicBroadcast, NonZero_BroadcastExplicit,
        ::testing::Combine(
            ::testing::ValuesIn(broadcastExplicitTestParams),
            ::testing::Values(ngraph::element::f16, ngraph::element::f32, ngraph::element::i32),
            ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)));

}  // namespace
