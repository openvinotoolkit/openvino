// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/reshape_transformation.hpp"

#include <memory>
#include <tuple>
#include <vector>
#include <string>
#include <ie_core.hpp>

#include "ngraph_functions/builders.hpp"
#include <transformations/init_node_info.hpp>


namespace LayerTestsDefinitions {

std::string ReshapeTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShapes, targetDevice, params, version) = obj.param;

    return getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version);
}

void ReshapeTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();
    const auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ConfigurePlugin(version);

    const float k = 50.f;

    const auto input = std::make_shared<ngraph::opset1::Parameter>(precision, ngraph::Shape(inputShape));
    const auto fakeQuantizeOnActivations = ngraph::builder::makeFakeQuantize(
        input, precision, 256ul, { 1ul },
        { 0.f }, { 255.f / k }, { 0.f }, { 255.f / k });

    const auto weights = ngraph::opset1::Constant::create(
        precision,
        ngraph::Shape{ inputShape[1], inputShape[1], 1, 1 },
        std::vector<float>(inputShape[1] * inputShape[1], 1));

    const auto convolution = std::make_shared<ngraph::opset1::Convolution>(
        fakeQuantizeOnActivations == nullptr ? input : fakeQuantizeOnActivations,
        ngraph::builder::makeFakeQuantize(weights, precision, 256ul, { 1ul }, { -128.f / k }, { 127.f / k }, { -128.f / k }, { 127.f / k }),
        ngraph::Strides{ 1, 1 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::CoordinateDiff{ 0, 0 },
        ngraph::Strides{ 1, 1 });

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(convolution)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { input }, "ReshapeTransformation");

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void ReshapeTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    std::tie(netPrecision, inputShape, targetDevice, params, version) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("ScaleShift", outputLayer->type);

    const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
    if (params.updatePrecisions) {
        checkPrecisions(
            *layer,
            { { InferenceEngine::Precision::U8 }, { InferenceEngine::Precision::I8 } },
            { getDeviceInternalPrecision(netPrecision) });
    } else {
        checkPrecisions(*layer, netPrecision);
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ReshapeTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
