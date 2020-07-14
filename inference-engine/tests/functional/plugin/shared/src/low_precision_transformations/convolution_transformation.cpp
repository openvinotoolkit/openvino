// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/convolution_transformation.hpp"

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


namespace LayerTestsDefinitions {

std::string ConvolutionTransformation::getTestCaseName(testing::TestParamInfo<ConvolutionTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    bool fqOnActivations;
    bool fqOnWeights;
    std::tie(netPrecision, inputShapes, targetDevice, params, fqOnActivations, fqOnWeights) = obj.param;

    std::ostringstream result;
    result << netPrecision.name() << "_" << targetDevice << "_" << toString(params) <<
        (fqOnActivations ? "" : "_noFqOnActivations") <<
        (fqOnWeights ? "" : "_noFqOnWeights");
    return result.str();
}

void ConvolutionTransformation::SetUp() {
    threshold = 0.1f;

    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    bool fqOnActivations;
    bool fqOnWeights;
    std::tie(netPrecision, inputShape, targetDevice, params, fqOnActivations, fqOnWeights) = this->GetParam();
    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const float k = 50.f;

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fakeQuantizeOnActivations = fqOnActivations ?
        ngraph::builder::makeFakeQuantize(
            input, precision, 256ul, { 1ul },
            { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k }) :
        nullptr;

    const auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
        std::vector<float>(inputShape[1] * inputShape[1], 1));

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        fakeQuantizeOnActivations == nullptr ? input : fakeQuantizeOnActivations,
        fqOnWeights ?
            ngraph::builder::makeFakeQuantize(
                weights, precision, 256ul, { 1ul },
                { -128.f / k }, { 127.f / k }, { -128.f / k }, { 127.f / k }) :
            weights->output(0),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(convolution)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { input }, "ConvolutionTransformation");

    validate();
}

void ConvolutionTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    bool fqOnActivations;
    bool fqOnWeights;
    std::tie(netPrecision, inputShape, targetDevice, params, fqOnActivations, fqOnWeights) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ(fqOnActivations & fqOnWeights ? "ScaleShift" : "Convolution", outputLayer->type);

    if (fqOnActivations & fqOnWeights) {
        const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
        if (params.updatePrecisions) {
            checkPrecisions(
                *layer,
                { { InferenceEngine::Precision::U8 }, { InferenceEngine::Precision::I8 } },
                { getDeviceInternalPrecision(netPrecision) });
        } else {
            checkPrecisions(*layer, netPrecision);
        }
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ConvolutionTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
