// Copyright (C) 2020 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#include "low_precision_transformations/clamp_transformation.hpp"

#include <sstream>
#include <string>
#include <vector>
#include <ngraph/ngraph.hpp>

#include "ngraph_functions/low_precision_transformations/clamp_function.hpp"

namespace LayerTestsDefinitions {

std::string ClampTransformation::getTestCaseName(testing::TestParamInfo<ClampTransformationParams> obj) {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShapes;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ClampTransformationParam param;;
    std::tie(netPrecision, inputShapes, targetDevice, params, version, param) = obj.param;

    std::ostringstream result;
    result << getTestCaseNameByParams(netPrecision, inputShapes, targetDevice, params, version) << "_" <<
        param.fakeQuantize << "_" <<
        "_min=" << param.clampLowConst <<
        "_max=" << param.clampHighConst;
    return result.str();
}

void ClampTransformation::SetUp() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    InferenceEngine::details::LayerTransformation::Params params;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ClampTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, params, version, param) = this->GetParam();
    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    ConfigurePlugin(version);

    function = ngraph::builder::subgraph::ClampFunction::getOriginal(
        precision,
        inputShape,
        param.fakeQuantize,
        param.clampLowConst,
        param.clampHighConst);

    if (version == LptVersion::cnnNetwork) {
        validate();
    }
}

void ClampTransformation::validate() {
    InferenceEngine::Precision netPrecision;
    InferenceEngine::SizeVector inputShape;
    std::string targetDevice;
    InferenceEngine::details::LayerTransformation::Params tmp;
    LayerTestsUtils::LayerTransformation::LptVersion version;
    ClampTransformationParam param;
    std::tie(netPrecision, inputShape, targetDevice, tmp, version, param) = this->GetParam();
    auto precision = FuncTestUtils::PrecisionUtils::convertIE2nGraphPrc(netPrecision);

    const auto params = LayerTestsUtils::LayerTransformationParamsNGraphFactory::createParamsU8I8();
    const InferenceEngine::CNNNetwork network = transform(toCNNNetwork(params));

    IE_SUPPRESS_DEPRECATED_START

    InferenceEngine::OutputsDataMap outputs = network.getOutputsInfo();
    EXPECT_EQ(1, outputs.size());

    std::map<std::string, InferenceEngine::DataPtr>::iterator it = outputs.begin();
    const InferenceEngine::CNNLayerPtr outputLayer = getCreatorLayer(it->second).lock();
    EXPECT_TRUE(outputLayer != nullptr);
    EXPECT_EQ("Clamp", outputLayer->type);

    const InferenceEngine::CNNLayerPtr layer = InferenceEngine::details::CNNNetworkHelper::getParent(*outputLayer);
    checkPrecisions(*layer, toCNNNetwork(precision));

    IE_SUPPRESS_DEPRECATED_END
}

TEST_P(ClampTransformation, CompareWithRefImpl) {
    Run();
};

} // namespace LayerTestsDefinitions
