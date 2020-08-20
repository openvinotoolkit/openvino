// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/permute_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>

#include <ie_core.hpp>

#include "common_test_utils/common_utils.hpp"
#include "functional_test_utils/plugin_cache.hpp"
#include "functional_test_utils/layer_test_utils.hpp"
#include "functional_test_utils/blob_utils.hpp"
#include "ngraph_functions/pass/convert_prc.hpp"
#include "ngraph_functions/builders.hpp"
#include "low_precision_transformations/network_helper.hpp"

namespace LayerTestsDefinitions {

std::string PermuteTransformation::getTestCaseName(testing::TestParamInfo<PermuteTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    std::string targetDevice;
    PermuteTransformationTestValues testValues;
    std::tie(netPrecision, targetDevice, testValues) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << targetDevice << "_" << toString(testValues.params) <<
        testValues.expected.permutePrecision << "_" <<
        testValues.actual.fqOutputLowIntervals.size() << "_" <<
        testValues.actual.fqOutputHighIntervals.size();
    return result.str();
}

void PermuteTransformation::SetUp() {
    InferenceEngine::Precision netPrecision;
    PermuteTransformationTestValues testValues;
    std::tie(netPrecision, targetDevice, testValues) = this->GetParam();
    const auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    {
        const auto input1 = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(testValues.shape));
        input1->set_friendly_name("input");

        const auto fakeQuantize = ngraph::builder::makeFakeQuantize(
            input1,
            precision,
            256ul,
            { 1, testValues.actual.fqOutputHighIntervals.size(), 1, 1 },
            testValues.actual.fqInputLowIntervals,
            testValues.actual.fqInputHighIntervals,
            testValues.actual.fqOutputLowIntervals,
            testValues.actual.fqOutputHighIntervals);

        const std::vector<size_t> kernel = { 3, 3 };
        const std::vector<size_t> stride = { 1, 1 };
        const std::vector<size_t> padBegin = { 0, 0 };
        const std::vector<size_t> padEnd = { 0, 0 };
        const ngraph::op::PadType padType = ngraph::op::PadType::NOTSET;
        const ngraph::op::RoundingType roundingType = ngraph::op::RoundingType::FLOOR;

        const auto maxPool = std::make_shared<ngraph::opset1::MaxPool>(
            fakeQuantize,
            stride,
            padBegin,
            padEnd,
            kernel,
            roundingType,
            padType);

        const auto transpose = std::make_shared<ngraph::opset1::Transpose>(
            maxPool,
            ngraph::opset1::Constant::create(ngraph::element::i64, ngraph::Shape{ testValues.order.size() }, testValues.order));
        transpose->set_friendly_name("transpose");

        ngraph::ResultVector results{ std::make_shared<ngraph::opset1::Result>(transpose) };
        function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector{ input1 }, "PermuteTransformation");
    }

    validate();
}

IE_SUPPRESS_DEPRECATED_START

void fillFromDequantizationLayer(
    const InferenceEngine::CNNLayer& dequantizationLayer,
    std::vector<float>& dequantizationScales,
    std::vector<float>& dequantizationShifts) {

    if (dequantizationLayer.type != "ScaleShift") {
        THROW_IE_EXCEPTION << "unexpected dequantization layer type " << dequantizationLayer.type;
    }

    InferenceEngine::CNNLayerPtr dequantizationLayerPtr = std::make_shared<InferenceEngine::CNNLayer>(dequantizationLayer);
    InferenceEngine::Blob::Ptr weightsBlob = InferenceEngine::details::CNNNetworkHelper::getBlob(dequantizationLayerPtr, "weights");
    const auto weightsBuffer = InferenceEngine::details::CNNNetworkHelper::getFloatData(weightsBlob);

    InferenceEngine::Blob::Ptr shiftsBlob = InferenceEngine::details::CNNNetworkHelper::getBlob(dequantizationLayerPtr, "biases");
    const auto shiftsBuffer = InferenceEngine::details::CNNNetworkHelper::getFloatData(shiftsBlob);

    const size_t inputCannelsCount = InferenceEngine::details::CNNNetworkHelper::getInputChannelsCount(dequantizationLayer);
    dequantizationScales.resize(inputCannelsCount);
    dequantizationShifts.resize(inputCannelsCount);
    for (size_t channel = 0; channel < inputCannelsCount; ++channel) {
        dequantizationScales[channel] = (weightsBlob->size() == 1ul) ? weightsBuffer.get()[0] : weightsBuffer.get()[channel];
        dequantizationShifts[channel] = (shiftsBlob->size() == 1ul) ? shiftsBuffer.get()[0] : shiftsBuffer.get()[channel];
    }
}

void PermuteTransformation::validate() {
    InferenceEngine::Precision netPrecision;
    PermuteTransformationTestValues testValues;
    std::tie(netPrecision, targetDevice, testValues) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(testValues.params);

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);

    InferenceEngine::CNNLayerPtr permute;
    if (testValues.expected.scales || testValues.expected.shifts) {
        EXPECT_EQ("ScaleShift", outputLayer->type);

        std::vector<float> dequantizationScales;
        std::vector<float> dequantizationShifts;
        fillFromDequantizationLayer(*outputLayer, dequantizationScales, dequantizationShifts);

        if (testValues.expected.scales) {
            ASSERT_TRUE(std::all_of(dequantizationScales.begin(), dequantizationScales.end(), [](const float value) { return value != 0.f; }));
        }

        if (testValues.expected.shifts) {
            ASSERT_TRUE(std::all_of(dequantizationShifts.begin(), dequantizationShifts.end(), [](const float value) { return value != 0.f; }));
        }

        permute = getCreatorLayer(outputLayer->insData[0].lock()).lock();
    } else {
        permute = outputLayer;
    }
    EXPECT_EQ("Permute", permute->type);

    const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
    checkPrecisions(*layer, testValues.expected.permutePrecision);
}

IE_SUPPRESS_DEPRECATED_END

TEST_P(PermuteTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
