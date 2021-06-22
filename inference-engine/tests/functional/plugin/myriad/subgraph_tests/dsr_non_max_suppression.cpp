// Copyright (C) 2018-2021 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "dsr_tests_common.hpp"

#include <vpu/ngraph/operations/dynamic_shape_resolver.hpp>
#include <ngraph/op/non_max_suppression.hpp>

namespace {

using namespace LayerTestsUtils::vpu;

using DataType = ngraph::element::Type_t;
using DataDims = ngraph::Shape;

struct NonMaxSuppressionTestCase {
    size_t numBatches, numBoxes, upperBoundNumBoxes, numClasses, maxOutputBoxesPerClass;
    float iouThreshold, scoreThreshold, softNmsSigma;
};

using Parameters = std::tuple<
    DataType,
    DataType,
    NonMaxSuppressionTestCase,
    LayerTestsUtils::TargetDevice
>;

class DSR_NonMaxSuppression : public testing::WithParamInterface<Parameters>,
                              public DSR_TestsCommon {
public:
    static std::string getTestCaseName(const testing::TestParamInfo<Parameters> &obj) {
        DataType floatType, integerType;
        NonMaxSuppressionTestCase nmsSetup;
        LayerTestsUtils::TargetDevice targetDevice;
        std::tie(floatType, integerType, nmsSetup, targetDevice) = obj.param;

        std::ostringstream result;
        result << "FT=" << floatType << "_";
        result << "IT=" << integerType << "_";
        result << "NBatches=" << nmsSetup.numBatches << "_";
        result << "NBoxes=" << nmsSetup.numBoxes << "_";
        result << "UBNBoxes=" << nmsSetup.upperBoundNumBoxes << "_";
        result << "NC=" << nmsSetup.numClasses << "_";
        result << "MaxB=" << nmsSetup.maxOutputBoxesPerClass << "_";
        result << "IOU=" << nmsSetup.iouThreshold << "_";
        result << "ScoreT=" << nmsSetup.scoreThreshold << "_";
        result << "Sigma=" << nmsSetup.softNmsSigma << "_";

        return result.str();
    }

protected:
    std::shared_ptr<ngraph::Node> createTestedOp() override {
        const auto& parameters = GetParam();
        const auto& floatType = std::get<0>(parameters);
        const auto& integerType = std::get<1>(parameters);
        const auto& nmsSetup = std::get<2>(parameters);
        targetDevice = std::get<3>(parameters);

        const auto boxes = createInputSubgraphWithDSR(floatType, DataShapeWithUpperBound{ {nmsSetup.numBatches, nmsSetup.numBoxes, 4},
            {nmsSetup.numBatches, nmsSetup.upperBoundNumBoxes, 4}});
        const auto scores = createInputSubgraphWithDSR(floatType, DataShapeWithUpperBound{ {nmsSetup.numBatches, nmsSetup.numClasses, nmsSetup.numBoxes},
            {nmsSetup.numBatches, nmsSetup.numClasses, nmsSetup.upperBoundNumBoxes}});

        const auto maxOutputBoxesPerClass = ngraph::opset5::Constant::create(integerType, ngraph::Shape{}, {nmsSetup.maxOutputBoxesPerClass});
        const auto iouThreshold = ngraph::opset5::Constant::create(floatType, ngraph::Shape{}, {nmsSetup.iouThreshold});
        const auto scoreThreshold = ngraph::opset5::Constant::create(floatType, ngraph::Shape{}, {nmsSetup.scoreThreshold});
        const auto softNmsSigma = ngraph::opset5::Constant::create(floatType, ngraph::Shape{}, {nmsSetup.softNmsSigma});

        return std::make_shared<ngraph::op::v5::NonMaxSuppression>(
            boxes, scores, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, softNmsSigma,
            ngraph::op::v5::NonMaxSuppression::BoxEncodingType::CENTER, false);
    }

    void SetUp() override {
            SetRefMode(LayerTestsUtils::RefMode::INTERPRETER);
            configuration[InferenceEngine::MYRIAD_DETECT_NETWORK_BATCH] = CONFIG_VALUE(NO);
            const auto testedOp = createTestedOp();
            const auto identity_0 = std::make_shared<ngraph::opset5::Multiply>(testedOp->output(0),
                ngraph::opset5::Constant::create(testedOp->output(0).get_element_type(), ngraph::Shape{1}, {1}));
            const auto identity_2 = std::make_shared<ngraph::opset5::Multiply>(testedOp->output(2),
                ngraph::opset5::Constant::create(testedOp->output(2).get_element_type(), ngraph::Shape{1}, {1}));
            function = std::make_shared<ngraph::Function>(
                ngraph::OutputVector{identity_0, identity_2},
                m_parameterVector);
    }
};

TEST_P(DSR_NonMaxSuppression, CompareWithReference) {
    Run();
}

INSTANTIATE_TEST_SUITE_P(smoke_DynamicNonMaxSupression, DSR_NonMaxSuppression,
    ::testing::Combine(
        ::testing::Values(
            ngraph::element::f16,
            ngraph::element::f32),
        ::testing::Values(
            ngraph::element::i32,
            ngraph::element::i64),
        ::testing::Values(
            // numBatches, numBoxes, upperBoundNumBoxes, numClasses, maxOutputBoxesPerClass, iouThreshold, scoreThreshold, softNmsSigma
            NonMaxSuppressionTestCase{1, 5, 10, 5, 10, 0., 0.},
            NonMaxSuppressionTestCase{2, 20, 100, 5, 10, 0., 0.},
            NonMaxSuppressionTestCase{3, 3, 10, 5, 2, 0.5, 0.},
            NonMaxSuppressionTestCase{1, 200, 1000, 1, 2000, 0.5, 0.}),
        ::testing::Values(CommonTestUtils::DEVICE_MYRIAD)),
    DSR_NonMaxSuppression::getTestCaseName);

}  // namespace
