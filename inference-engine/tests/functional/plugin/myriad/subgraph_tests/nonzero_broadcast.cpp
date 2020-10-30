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
                                       virtual public LayerTestsUtils::LayerTestsCommon {
protected:
    size_t getDynamicAxis(const DataShape& shapeA, const DataShape& shapeB) const {
        size_t res = 0;
        while (shapeA[res] == shapeB[res]) {
            res++;
        }
        return res;
    }

    void prepareBroadcastInputs() {
        SetRefMode(LayerTestsUtils::RefMode::CONSTANT_FOLDING);

        const auto& parameters = GetParam();
        const auto& broadcastParams = std::get<0>(parameters);
        const auto& tensorType = std::get<1>(parameters);
        targetDevice = std::get<2>(GetParam());

        const auto& upperBoundShape = broadcastParams.targetShape.upperBoundShape;
        const auto& realShape = broadcastParams.targetShape.shape;

        const auto dynamicAxis = getDynamicAxis(upperBoundShape, realShape);

        m_param = std::make_shared<ngraph::opset5::Parameter>(tensorType, TensorShape{upperBoundShape[dynamicAxis]});
        m_nonZero = std::make_shared<ngraph::opset5::NonZero>(m_param);
        const auto shapeOfNonZero = std::make_shared<ngraph::opset5::ShapeOf>(m_nonZero);
        const auto numNonZeros = std::make_shared<ngraph::opset5::Gather>(
            shapeOfNonZero,
            ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {1}),
            ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{1}, {0}));

        m_broadcastTargetShape = numNonZeros;

        if (dynamicAxis > 0) {
            m_broadcastTargetShape = std::make_shared<ngraph::opset5::Concat>(
                ngraph::NodeVector{
                    ngraph::opset5::Constant::create(
                        ngraph::element::i64,
                        ngraph::Shape{dynamicAxis},
                        std::vector<size_t>{upperBoundShape.begin(), upperBoundShape.begin() + dynamicAxis}),
                    m_broadcastTargetShape},
                0);
        }

        if (dynamicAxis < upperBoundShape.size() - 1) {
            m_broadcastTargetShape = std::make_shared<ngraph::opset5::Concat>(
                ngraph::NodeVector{
                    m_broadcastTargetShape,
                    ngraph::opset5::Constant::create(
                        ngraph::element::i64,
                        ngraph::Shape{upperBoundShape.size() - dynamicAxis - 1},
                        std::vector<size_t>{upperBoundShape.begin() + dynamicAxis + 1, upperBoundShape.end()})},
                0);
        }

        m_broadcastInput = ngraph::builder::makeConstant(tensorType, ngraph::Shape{broadcastParams.inputShape}, std::vector<int64_t>{}, true);
    }

    void SetUp() override {
        prepareBroadcastInputs();

        const auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(m_broadcastInput, m_broadcastTargetShape, ngraph::op::BroadcastType::BIDIRECTIONAL);

        function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{broadcast, m_nonZero},
            ngraph::ParameterVector{m_param},
            "NonZero-Broadcast");
    }

    InferenceEngine::Blob::Ptr GenerateInput(const InferenceEngine::InputInfo& info) const override {
        // We emulate dynamic shape through the number of non-zeros in NonZero input tensor
        const auto& broadcastParams = std::get<0>(GetParam());
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

protected:
    std::shared_ptr<ngraph::Node> m_broadcastInput;
    std::shared_ptr<ngraph::Node> m_broadcastTargetShape;
    std::shared_ptr<ngraph::opset5::NonZero> m_nonZero;
    std::shared_ptr<ngraph::opset5::Parameter> m_param;
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
    void SetUp() override {
        prepareBroadcastInputs();

        const auto& axesMapping = std::get<0>(GetParam()).axesMapping;
        const auto axesMappingConst = ngraph::opset5::Constant::create(ngraph::element::i64, ngraph::Shape{axesMapping.size()}, axesMapping);

        const auto broadcast = std::make_shared<ngraph::opset5::Broadcast>(m_broadcastInput, m_broadcastTargetShape, axesMappingConst);

        function = std::make_shared<ngraph::Function>(
            ngraph::NodeVector{broadcast, m_nonZero},
            ngraph::ParameterVector{m_param},
            "NonZero-Broadcast");
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
