// Copyright (C) 2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/fully_connected_transformation.hpp"

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
#include "low_precision_transformations/network_helper.hpp"
#include "ngraph_functions/builders.hpp"

namespace LayerTestsDefinitions {

std::string FullyConnectedTransformation::getTestCaseName(testing::TestParamInfo<LayerTestsUtils::LayerTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShapes, targetDevice, params) = obj.param;

    std::ostringstream result;
    result << inputShapes.size() << "D_" << netPrecision.name() << "_" << targetDevice << "_" << toString(params);
    return result.str();
}

void FullyConnectedTransformation::SetUp() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();
    auto ngPrecision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto paramNode = std::make_shared<ngraph::opset1::Parameter>(ngPrecision, ngraph::Shape(inputShape));
    const std::vector<size_t> constShapes(inputShape.size(), 1ul);
    const auto fakeQuantizeOnAcitvations = ngraph::builder::makeFakeQuantize(
        paramNode, ngPrecision, 256ul, constShapes,
        { 0.f }, { 255.f / 4.f }, { 0.f }, { 255.f / 4.f });
    fakeQuantizeOnAcitvations->set_friendly_name("fakeQuantizeOnAcitvations");

    auto weightsConst = std::make_shared<ngraph::op::Constant>(
        ngPrecision,
        ngraph::Shape { inputShape[inputShape.size() - 1ul], inputShape[inputShape.size() - 1ul] },
        std::vector<float>({ 1.f }));
    const auto fakeQuantizeOnWeights = ngraph::builder::makeFakeQuantize(
        weightsConst, ngPrecision, 256ul, { 1ul, 1ul },
        { -128.f / 8.f }, { 127.f / 8.f }, { -128.f / 8.f }, { 127.f / 8.f });
    fakeQuantizeOnWeights->set_friendly_name("fakeQuantizeOnWeights");

    const std::shared_ptr<ngraph::opset1::MatMul> fullyConnected = std::make_shared<ngraph::opset1::MatMul>(
        fakeQuantizeOnAcitvations->output(0),
        fakeQuantizeOnWeights->output(0),
        false,
        false);
    fullyConnected->set_friendly_name("fullyConnected");

    ngraph::ResultVector results {std::make_shared<ngraph::opset1::Result>(fullyConnected)};
    function = std::make_shared<ngraph::Function>(results, ngraph::ParameterVector { paramNode }, "FullyConnectedTransformation");

    // TODO: move to some another place
    validate();
}

void FullyConnectedTransformation::validate() {
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::Precision netPrecision;
    InferenceEngine::details::LayerTransformation::Params params;
    std::tie(netPrecision, inputShape, targetDevice, params) = this->GetParam();

    const InferenceEngine::CNNNetwork network = transform(params);

    IE_SUPPRESS_DEPRECATED_START

    const InferenceEngine::CNNLayerPtr fullyConnected = InferenceEngine::details::CNNNetworkHelper::getLayer(network, "fullyConnected_original");
    EXPECT_NE(nullptr, fullyConnected) << "fullyConnected_original was not found, transformation was not handled";
    EXPECT_EQ("FullyConnected", fullyConnected->type);

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    for (const auto it : outputs) {
        const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it.second).lock();
        EXPECT_TRUE(outputLayer != nullptr);
        EXPECT_EQ("ScaleShift", outputLayer->type);

        const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
        if (params.updatePrecisions) {
            const bool asymmetricQuantizationOnData = std::all_of(
                params.precisionsOnActivations.begin(),
                params.precisionsOnActivations.end(),
                [](const InferenceEngine::Precision precision) { return precision != InferenceEngine::Precision::U8; });

            const bool asymmetricQuantizationOnWeights = std::all_of(
                params.precisionsOnWeights.begin(),
                params.precisionsOnWeights.end(),
                [](const InferenceEngine::Precision precision) { return precision != InferenceEngine::Precision::I8; });

            checkPrecisions(
                *layer,
                { { params.precisionsOnActivations[0] }, { params.precisionsOnWeights[0] }, { netPrecision } },
                { getDeviceInternalPrecision(netPrecision) },
                asymmetricQuantizationOnData,
                asymmetricQuantizationOnWeights);
        } else {
            checkPrecisions(*layer, netPrecision);
        }
    }

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(FullyConnectedTransformation, CompareWithRefImpl) {
    Run();
};

}  // namespace LayerTestsDefinitions
